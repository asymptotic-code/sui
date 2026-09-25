// Copyright (c) The Diem Core Contributors
// Copyright (c) The Move Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use move_command_line_common::files::{
    FileHash, MOVE_COMPILED_EXTENSION, extension_equals, find_filenames, find_move_filenames,
};
use move_compiler::command_line::DEFAULT_OUTPUT_DIR;
use move_compiler::editions::Edition;
use move_compiler::{diagnostics::filter::empty_filter_scope, shared::PackageConfig};
use move_core_types::account_address::AccountAddress;
use move_symbol_pool::Symbol;
use std::fs::File;
use std::str::FromStr;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use treeline::Tree;

use crate::lock_file::schema::ManagedPackage;
use crate::package_hooks::{PackageIdentifier, custom_resolve_pkg_id};
use crate::source_package::parsed_manifest as PM;
use crate::{
    BuildConfig,
    source_package::{
        layout::SourcePackageLayout,
        manifest_parser::parse_move_manifest_from_file,
        parsed_manifest::{
            FileName, NamedAddress, PackageDigest, PackageName, SourceManifest, SubstOrRename,
        },
    },
};

use super::{
    dependency_cache::DependencyCache, dependency_graph as DG, digest::compute_digest, local_path,
    resolving_table::ResolvingTable,
};

/// The graph after resolution in which all named addresses have been assigned a value.
///
/// Named addresses can be assigned values in a couple different ways:
/// 1. They can be assigned a value in the declaring package. In this case the value of that
///    named address will always be that value.
/// 2. Can be left unassigned in the declaring package. In this case it can receive its value
///    through unification across the package graph.
///
/// Named addresses can also be renamed in a package and will be re-exported under thes new names in
/// this case.
#[derive(Debug, Clone)]
pub struct ResolvedGraph {
    pub graph: DG::DependencyGraph,
    /// Build options
    pub build_options: BuildConfig,
    /// A mapping of package name to its resolution
    pub package_table: PackageTable,
}

// rename_to => (from_package_name, from_address_name)
pub type Renaming = BTreeMap<NamedAddress, (PackageName, NamedAddress)>;
pub type ResolvedTable = BTreeMap<NamedAddress, AccountAddress>;
type PackageTable = BTreeMap<PackageName, Package>;

#[derive(Debug, Clone)]
pub struct Package {
    /// Source manifest for this package
    pub source_package: SourceManifest,
    /// Where this package is located on the filesystem
    pub package_path: PathBuf,
    /// The renaming of addresses performed by this package
    pub renaming: Renaming,
    /// The mapping of addresses that are in scope for this package.
    pub resolved_table: ResolvedTable,
    /// The digest of the contents of all source files and manifest under the package root
    pub source_digest: PackageDigest,
}

impl ResolvedGraph {
    pub fn resolve<Progress: Write>(
        graph: DG::DependencyGraph,
        build_options: BuildConfig,
        dependency_cache: &mut DependencyCache,
        chain_id: Option<String>,
        progress_output: &mut Progress,
    ) -> Result<ResolvedGraph> {
        let mut package_table = PackageTable::new();
        let mut resolving_table = ResolvingTable::new();

        let dep_mode = if build_options.dev_mode {
            DG::DependencyMode::DevOnly
        } else {
            DG::DependencyMode::Always
        };

        let relocations =
            zero_address_relocations(&graph, &build_options, dependency_cache, progress_output)?;

        // Resolve transitive dependencies in reverse topological order so that a package's
        // dependencies get resolved before it does.
        for pkg_id in graph.topological_order().into_iter().rev() {
            // Skip dev-mode packages if not in dev-mode.
            if !(build_options.dev_mode || graph.always_deps.contains(&pkg_id)) {
                continue;
            }

            // Make sure the package is available locally.
            let package_path = if pkg_id == graph.root_package_id {
                graph.root_path.clone()
            } else {
                let pkg = &graph.package_table[&pkg_id];
                dependency_cache
                    .download_and_update_if_remote(pkg_id, &pkg.kind, progress_output)
                    .with_context(|| format!("Fetching '{pkg_id}'"))?;
                graph.root_path.join(local_path(&pkg.kind))
            };

            let mut resolved_pkg = Package::new(package_path, &build_options)
                .with_context(|| format!("Resolving package '{pkg_id}'"))?;

            // Check dependencies package names from manifest are consistent with ther names
            // in parent (this) manifest. We do this check only for local and git
            // dependencies as we assume custom dependencies might not have a user-defined
            // name.
            for (dep_name, dep) in &resolved_pkg.source_package.dependencies {
                match dep {
                    PM::Dependency::External(_) => continue,
                    PM::Dependency::Internal(internal) => {
                        if let PM::DependencyKind::OnChain(_) = internal.kind {
                            continue;
                        }
                        dependency_cache
                            .download_and_update_if_remote(
                                *dep_name,
                                &internal.kind,
                                progress_output,
                            )
                            .with_context(|| format!("Fetching '{dep_name}'"))?;

                        let dep_path = &resolved_pkg.package_path.join(local_path(&internal.kind));
                        let dep_manifest = parse_move_manifest_from_file(dep_path)?;
                        let rename_from = package_rename(internal.subst.as_ref(), dep_name);
                        if !dep_name_matches_manifest(
                            dep_name,
                            rename_from,
                            &dep_manifest,
                            dep_path,
                        ) {
                            bail!(
                                "Name of dependency '{}' does not match dependency's package name '{}'",
                                dep_name,
                                dep_manifest.package.name
                            )
                        }
                    }
                };
            }

            let pkg_name = resolved_pkg.source_package.package.name;

            resolved_pkg
                .define_addresses_in_package(
                    &mut resolving_table,
                    &chain_id,
                    relocations.get(&pkg_id).copied(),
                )
                .with_context(|| format!("Resolving addresses for '{pkg_name}'"))?;

            for (dep_id, dep, _pkg) in graph.immediate_dependencies(pkg_id, dep_mode) {
                let dep_name = dep.dep_name;
                resolved_pkg
                    .process_dependency(dep_id, dep, &package_table, &mut resolving_table)
                    .with_context(|| {
                        format!("Processing dependency '{dep_name}' of '{pkg_name}'")
                    })?;
            }

            package_table.insert(pkg_id, resolved_pkg);
        }

        // Add additional addresses to all package resolution tables.
        for (name, addr) in &build_options.additional_named_addresses {
            let name = NamedAddress::from(name.as_str());
            for pkg in package_table.keys() {
                resolving_table
                    .define((*pkg, name), Some(*addr))
                    .with_context(|| {
                        format!("Adding additional address '{name}' to package '{pkg}'")
                    })?;
            }
        }

        let root_package = &package_table[&graph.root_package_id];

        // Add dev addresses, but only for the root package
        if build_options.dev_mode {
            let mut addr_to_name_mapping = BTreeMap::new();
            for (name, addr) in resolving_table.bindings(graph.root_package_id) {
                if let Some(addr) = addr {
                    addr_to_name_mapping
                        .entry(*addr)
                        .or_insert_with(Vec::new)
                        .push(name);
                };
            }

            for (name, addr) in root_package
                .source_package
                .dev_address_assignments
                .iter()
                .flatten()
            {
                let root_dev_addr = (graph.root_package_id, *name);
                if !resolving_table.contains(root_dev_addr) {
                    bail!(
                        "Found unbound dev address assignment '{} = 0x{}' in root package '{}'. \
                         Dev addresses cannot introduce new named addresses",
                        name,
                        addr.short_str_lossless(),
                        graph.root_package_name,
                    );
                }

                resolving_table
                    .define(root_dev_addr, Some(*addr))
                    .with_context(|| {
                        format!(
                            "Unable to resolve named address '{}' in package '{}' when resolving \
                             dependencies in dev mode",
                            name, graph.root_package_name,
                        )
                    })?;

                if let Some(conflicts) = addr_to_name_mapping.insert(*addr, vec![*name]) {
                    bail!(
                        "Found non-unique dev address assignment '{name} = 0x{addr}' in root \
                         package '{pkg}'. Dev address assignments must not conflict with any other \
                         assignments in order to ensure that the package will compile with any \
                         possible address assignment. \
                         Assignment conflicts with previous assignments: {conflicts} = 0x{addr}",
                        name = name,
                        addr = addr.short_str_lossless(),
                        pkg = graph.root_package_name,
                        conflicts = conflicts
                            .iter()
                            .map(NamedAddress::as_str)
                            .collect::<Vec<_>>()
                            .join(", "),
                    )
                }
            }
        }

        // Now that all address unification has happened, individual package resolution tables can
        // be unified.
        for pkg in package_table.values_mut() {
            pkg.finalize_address_resolution(&resolving_table)
                .with_context(|| {
                    format!(
                        "Unresolved addresses found. To fix this, add an entry for each unresolved \
                         address to the [addresses] section of {}/Move.toml: e.g.,\n\n\
                         \
                         [addresses]\n\
                         std = \"0x1\"\n\n\
                         \
                         Alternatively, you can also define [dev-addresses] and call with the -d \
                         flag",
                        graph.root_path.display()
                    )
                })?;
        }

        Ok(ResolvedGraph {
            graph,
            build_options,
            package_table,
        })
    }

    pub fn root_package(&self) -> PackageIdentifier {
        self.graph.root_package_id
    }

    pub fn get_package(&self, name: PackageName) -> &Package {
        self.package_table.get(&name).unwrap()
    }

    /// Return the names of packages in this resolution graph in topological order.
    pub fn topological_order(&self) -> Vec<PackageName> {
        let mut order = self.graph.topological_order();
        if !self.build_options.dev_mode {
            order.retain(|pkg| self.graph.always_deps.contains(pkg));
        }
        order
    }

    fn print_info_dfs(&self, current_node: &PackageName, tree: &mut Tree<String>) -> Result<()> {
        let pkg = self.package_table.get(current_node).unwrap();

        for (name, addr) in &pkg.resolved_table {
            tree.push(Tree::root(format!(
                "{}:0x{}",
                name,
                addr.short_str_lossless()
            )));
        }

        for dep in pkg.immediate_dependencies(self) {
            let mut child = Tree::root(dep.to_string());
            self.print_info_dfs(&dep, &mut child)?;
            tree.push(child);
        }

        Ok(())
    }

    pub fn print_info(&self) -> Result<()> {
        let root = self.root_package();
        let mut tree = Tree::root(root.to_string());
        self.print_info_dfs(&root, &mut tree)?;
        println!("{}", tree);
        Ok(())
    }

    pub fn extract_named_address_mapping(
        &self,
    ) -> impl Iterator<Item = (NamedAddress, AccountAddress)> {
        self.package_table
            .get(&self.root_package())
            .expect("Failed to find root package in package table -- this should never happen")
            .resolved_table
            .clone()
            .into_iter()
    }

    pub fn file_sources(&self) -> BTreeMap<FileHash, (FileName, String)> {
        self.package_table
            .values()
            .flat_map(|rpkg| {
                rpkg.get_sources(&self.build_options)
                    .unwrap()
                    .iter()
                    .map(|fname| {
                        let contents = fs::read_to_string(fname.as_str()).unwrap();
                        let fhash = FileHash::new(&contents);
                        (fhash, (*fname, contents))
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .collect()
    }

    pub fn contains_renaming(&self) -> Option<PackageName> {
        // Make sure no renamings have been performed
        self.package_table
            .iter()
            .find_map(|(name, pkg)| (!pkg.renaming.is_empty()).then_some(*name))
    }

    // Extract a remapping for each package from "local (in package) named address" to the "name in
    // the root package resolve addresses table". If no renaming is performed across the package
    // graph than the resulting mapping will be the identity mapping for each package.
    pub fn root_renaming(
        &self,
    ) -> BTreeMap<PackageIdentifier, BTreeMap<NamedAddress, NamedAddress>> {
        let mut root_mapping = BTreeMap::new();
        let current_address_mapping = self
            .extract_named_address_mapping()
            .map(|(name, _)| (name, name))
            .collect();

        self.compute_root_renaming(
            self.root_package(),
            &current_address_mapping,
            BTreeMap::new(),
            &mut root_mapping,
        );

        root_mapping
    }

    fn compute_root_renaming(
        &self,
        current_node: PackageIdentifier,
        // Parent name -> root name
        parent_in_scope_names: &BTreeMap<NamedAddress, NamedAddress>,
        // Parent name -> child name
        remapping: BTreeMap<NamedAddress, NamedAddress>,
        // package id -> { package local name -> root name }
        root_renaming: &mut BTreeMap<PackageIdentifier, BTreeMap<NamedAddress, NamedAddress>>,
    ) {
        let pkg = self.package_table.get(&current_node).unwrap();
        let global_rename_for_pkg: BTreeMap<_, _> = parent_in_scope_names
            .iter()
            .filter_map(|(parent_name, global_name)| {
                let local_name = remapping.get(parent_name).unwrap_or(parent_name);
                if pkg.resolved_table.contains_key(local_name) {
                    Some((*local_name, *global_name))
                } else {
                    None
                }
            })
            .collect();

        let mut per_dep_renaming = BTreeMap::new();
        for (to, (from_pkg, from_name)) in &pkg.renaming {
            per_dep_renaming
                .entry(*from_pkg)
                .or_insert_with(BTreeMap::new)
                .insert(*to, *from_name);
        }

        for dep in pkg.immediate_dependencies(self) {
            self.compute_root_renaming(
                dep,
                &global_rename_for_pkg,
                per_dep_renaming.remove(&dep).unwrap_or(BTreeMap::new()),
                root_renaming,
            );
        }

        root_renaming.insert(current_node, global_rename_for_pkg);
    }
}

impl Package {
    fn new(package_path: PathBuf, config: &BuildConfig) -> Result<Package> {
        Ok(Package {
            source_package: parse_move_manifest_from_file(&package_path)?,
            source_digest: package_digest_for_config(&package_path, config)?,
            package_path,
            renaming: Renaming::new(),
            resolved_table: ResolvedTable::new(),
        })
    }

    /// Associates addresses with named packages in the `resolving_table`.
    /// Addresses may be pulled in from two sources:
    /// - The [addresses] section `Move.toml`.
    /// - Adresses (package IDs) in the `Move.lock` associated with published packages for `chain_id`.
    ///
    /// Addresses are pulled from the `Move.lock` only when a package is published or upgraded on-chain.
    /// Local builds only consult the `Move.toml` manifest.
    ///
    /// `relocation` moves the package's 0x0 addresses off 0x0 (see `zero_address_relocations`).
    fn define_addresses_in_package(
        &self,
        resolving_table: &mut ResolvingTable,
        chain_id: &Option<String>,
        relocation: Option<AccountAddress>,
    ) -> Result<()> {
        let pkg_id = custom_resolve_pkg_id(&self.source_package).with_context(|| {
            format!(
                "Resolving package name for '{}'",
                &self.source_package.package.name
            )
        })?;
        for (name, addr) in self.source_package.addresses.iter().flatten() {
            if *addr == Some(AccountAddress::ZERO) {
                // The address in the manifest is set to 0x0, meaning `name` is associated with 'this'
                // package. Published dependent package IDs are resolved by `chain_id` from the
                // `Move.lock` when a package is to be published or upgraded.
                if let Some(original_id) = self.resolve_original_id_from_lock(chain_id) {
                    let addr = AccountAddress::from_str(&original_id)?;
                    resolving_table.define((pkg_id, *name), Some(addr))?;
                    continue;
                }
                if let Some(addr) = relocation {
                    resolving_table.define((pkg_id, *name), Some(addr))?;
                    continue;
                }
            }
            resolving_table.define((pkg_id, *name), *addr)?;
        }
        Ok(())
    }

    fn resolve_original_id_from_lock(&self, chain_id: &Option<String>) -> Option<String> {
        let lock_file = self.package_path.join(SourcePackageLayout::Lock.path());
        let mut lock_file = File::open(lock_file).ok()?;
        let managed_packages = ManagedPackage::read(&mut lock_file).ok();
        managed_packages
            .and_then(|m| {
                let chain_id = chain_id.as_ref()?;
                m.into_iter().find(|(_, v)| v.chain_id == *chain_id)
            })
            .map(|(_, v)| v.original_published_id)
    }

    fn process_dependency(
        &mut self,
        dep_id: PackageName,
        dep: &DG::Dependency,
        package_table: &PackageTable,
        resolving_table: &mut ResolvingTable,
    ) -> Result<()> {
        let pkg_name = self.source_package.package.name;
        let pkg_id = custom_resolve_pkg_id(&self.source_package).with_context(|| {
            format!(
                "Resolving package name for '{}'",
                &self.source_package.package.name
            )
        })?;
        let dep_name = dep.dep_name;

        let mut dep_renaming = BTreeMap::new();

        for (to, subst) in dep.subst.iter().flatten() {
            match subst {
                SubstOrRename::Assign(addr) => {
                    resolving_table.define((pkg_id, *to), Some(*addr))?;
                }

                SubstOrRename::RenameFrom(from) => {
                    if !resolving_table.contains((dep_id, *from)) {
                        bail!(
                            "Tried to rename named address {0} from package '{1}', \
                             however {1} does not contain that address",
                            from,
                            dep_name,
                        )
                    }

                    if let Some((prev_dep, prev_from)) = self.renaming.insert(*to, (dep_id, *from))
                    {
                        bail!(
                            "Duplicate renaming of named address '{to}' in dependencies of \
                             '{pkg_name}'. Substituted with '{from}' from dependency '{dep_name}' \
                             and '{prev_from}' from dependency '{prev_dep}'.",
                        )
                    }

                    dep_renaming.insert(*from, *to);
                }

                SubstOrRename::PackageRename(from) => {
                    if !resolving_table.contains((dep_id, *from)) {
                        bail!(
                            "'{dep_name}' is declared with rename-from = \"{from}\", \
                             however '{dep_name}' does not contain that address",
                        )
                    }
                    dep_renaming.insert(*from, *to);
                }
            }
        }

        let bound_in_dep: Vec<_> = resolving_table
            .bindings(dep_id)
            .map(|(from, _)| from)
            .collect();

        let own_names = self.own_address_names();
        for from in bound_in_dep {
            let to = *dep_renaming.get(&from).unwrap_or(&from);
            if let Err(conflict) = resolving_table.unify((pkg_id, to), (dep_id, from)) {
                // A modern package (no [addresses]) sees only its own name and its direct
                // dependencies' names (move-package-alt `named_addresses`); the transitive names
                // the legacy scope also carries are a convenience. When two dependencies bring the
                // same transitive name at different addresses, that name is dropped from this
                // package's scope instead of failing the build -- the package cannot be using it,
                // or the new resolver would reject it too.
                if self.is_modern() && to != dep_name && !own_names.contains(&to) {
                    resolving_table.hide((pkg_id, to));
                    continue;
                }
                return Err(conflict);
            }
        }

        let Some(resolved_dep) = package_table.get(&dep_id) else {
            bail!(
                "Unable to find resolved information for dependency '{dep_name}' of \
                 '{pkg_name}'",
            );
        };

        if let Some(digest) = dep.digest
            && digest != resolved_dep.source_digest
        {
            bail!(
                "Source digest mismatch in dependency '{dep_name}' of '{pkg_name}'. \
                     Expected '{digest}' but got '{}'.",
                resolved_dep.source_digest
            )
        }

        Ok(())
    }

    /// A package written for the modern package system: a 2024-edition manifest with none of the
    /// legacy-only sections (move-package-alt's `try_load_legacy_manifest` rule) and no legacy
    /// `addr_subst` on a dependency. Older manifests keep the legacy address scope unchanged.
    fn is_modern(&self) -> bool {
        let pkg = &self.source_package;
        pkg.addresses.is_none()
            && pkg.dev_address_assignments.is_none()
            && pkg.dev_dependencies.is_empty()
            && pkg
                .package
                .edition
                .is_some_and(|e| e.edition.as_str() == "2024")
            && pkg.dependencies.values().all(|dep| match dep {
                PM::Dependency::Internal(internal) => internal
                    .subst
                    .iter()
                    .flatten()
                    .all(|(_, s)| matches!(s, SubstOrRename::PackageRename(_))),
                PM::Dependency::External(_) => true,
            })
    }

    /// The names this package binds itself: its package name and its declared addresses.
    fn own_address_names(&self) -> BTreeSet<NamedAddress> {
        let mut names: BTreeSet<NamedAddress> = self
            .source_package
            .addresses
            .iter()
            .flatten()
            .map(|(name, _)| *name)
            .collect();
        names.insert(self.source_package.package.name);
        names
    }

    fn finalize_address_resolution(&mut self, resolving_table: &ResolvingTable) -> Result<()> {
        let pkg_name = self.source_package.package.name;
        let pkg_id = custom_resolve_pkg_id(&self.source_package).with_context(|| {
            format!(
                "Resolving package name for '{}'",
                &self.source_package.package.name
            )
        })?;
        let mut unresolved_addresses = Vec::new();

        for (name, addr) in resolving_table.bindings(pkg_id) {
            match *addr {
                Some(addr) => {
                    self.resolved_table.insert(name, addr);
                }
                None => {
                    unresolved_addresses
                        .push(format!("  Named address '{name}' in package '{pkg_name}'"));
                }
            }
        }

        if !unresolved_addresses.is_empty() {
            bail!(
                "Unresolved addresses: [\n{}\n]",
                unresolved_addresses.join("\n"),
            )
        }

        Ok(())
    }

    pub fn immediate_dependencies(&self, graph: &ResolvedGraph) -> BTreeSet<PackageName> {
        let pkg_id = custom_resolve_pkg_id(&self.source_package)
            .with_context(|| {
                format!(
                    "Resolving package name for '{}'",
                    &self.source_package.package.name
                )
            })
            .unwrap();

        graph
            .graph
            .immediate_dependencies(
                pkg_id,
                if graph.build_options.dev_mode {
                    DG::DependencyMode::DevOnly
                } else {
                    DG::DependencyMode::Always
                },
            )
            .map(|(name, _, _)| name)
            .collect()
    }

    pub fn get_sources(&self, config: &BuildConfig) -> Result<Vec<FileName>> {
        let places_to_look = source_paths_for_config(&self.package_path, config);
        Ok(find_move_filenames(&places_to_look, false)?
            .into_iter()
            .map(FileName::from)
            .collect())
    }

    fn get_build_paths(package_path: &Path) -> Result<Vec<PathBuf>> {
        let mut places_to_look = Vec::new();
        let path = package_path.join(Path::new(DEFAULT_OUTPUT_DIR));
        if path.exists() {
            places_to_look.push(path);
        }
        Ok(places_to_look)
    }

    pub fn get_bytecodes(&self) -> Result<Vec<FileName>> {
        let path = Package::get_build_paths(&self.package_path)?;
        Ok(find_filenames(&path, |path| {
            extension_equals(path, MOVE_COMPILED_EXTENSION)
        })?
        .into_iter()
        .map(Symbol::from)
        .collect())
    }

    pub fn get_bytecodes_bytes(&self) -> Result<Vec<Vec<u8>>> {
        let mut ret = vec![];
        for path in self.get_bytecodes()? {
            let bytes = std::fs::read(path.to_string())?;
            ret.push(bytes);
        }

        Ok(ret)
    }

    pub fn compiler_config(&self, is_dependency: bool, config: &BuildConfig) -> PackageConfig {
        PackageConfig {
            is_dependency,
            flavor: self
                .source_package
                .package
                .flavor
                .or(config.default_flavor)
                .unwrap_or_default(),
            edition: self
                .source_package
                .package
                .edition
                .or(config.default_edition)
                .unwrap_or(Edition::LEGACY), // TODO require edition
            warning_filter: empty_filter_scope(),
        }
    }
}

/// The environment whose publication records (`Published.toml`) give a relocated package its
/// address: `MOVE_BUILD_ENV`, else `mainnet` (the `sui move build --build-env` counterpart).
fn build_env() -> String {
    std::env::var("MOVE_BUILD_ENV")
        .ok()
        .filter(|e| !e.trim().is_empty())
        .unwrap_or_else(|| "mainnet".to_string())
}

/// Packages that must leave 0x0, and the address each moves to.
///
/// Every legacy package binds its own address to 0x0, so two unpublished packages declaring the
/// same module (`deepbook::registry` and `bs_oracle::registry`) define it twice at 0x0.
/// move-package-alt never puts two packages at one address: a published dependency sits at its
/// `original-id` for the build environment, an unpublished one at a per-package dummy address.
/// Here only packages that actually collide move, so every build that works today keeps the
/// exact addresses it has. The root and, after it, packages local to the repository keep 0x0;
/// the others take their `Published.toml` `original-id` for `build_env()`, else a dummy address.
fn zero_address_relocations<Progress: Write>(
    graph: &DG::DependencyGraph,
    build_options: &BuildConfig,
    dependency_cache: &mut DependencyCache,
    progress_output: &mut Progress,
) -> Result<BTreeMap<PackageName, AccountAddress>> {
    let mut modules: BTreeMap<String, Vec<PackageName>> = BTreeMap::new();
    let mut paths: BTreeMap<PackageName, PathBuf> = BTreeMap::new();
    for pkg_id in graph.topological_order() {
        if !(build_options.dev_mode || graph.always_deps.contains(&pkg_id)) {
            continue;
        }
        let path = if pkg_id == graph.root_package_id {
            graph.root_path.clone()
        } else {
            let pkg = &graph.package_table[&pkg_id];
            if matches!(pkg.kind, PM::DependencyKind::OnChain(_)) {
                continue;
            }
            dependency_cache
                .download_and_update_if_remote(pkg_id, &pkg.kind, progress_output)
                .with_context(|| format!("Fetching '{pkg_id}'"))?;
            graph.root_path.join(local_path(&pkg.kind))
        };
        let Ok(manifest) = parse_move_manifest_from_file(&path) else {
            continue;
        };
        let zero: BTreeSet<String> = manifest
            .addresses
            .iter()
            .flatten()
            .filter(|(_, addr)| **addr == Some(AccountAddress::ZERO))
            .map(|(name, _)| name.to_string())
            .collect();
        if zero.is_empty() {
            continue;
        }
        for (addr, module) in module_decls(&path.join(SourcePackageLayout::Sources.path()), 8) {
            if zero.contains(&addr) {
                modules.entry(module).or_default().push(pkg_id);
            }
        }
        paths.insert(pkg_id, path);
    }

    let keeps_zero = |pkg: &PackageName| {
        if *pkg == graph.root_package_id {
            0
        } else if matches!(
            graph.package_table.get(pkg).map(|p| &p.kind),
            Some(PM::DependencyKind::Local(_))
        ) {
            1
        } else {
            2
        }
    };
    let env = build_env();
    let mut relocations = BTreeMap::new();
    for pkgs in modules.into_values() {
        let mut pkgs: Vec<PackageName> = pkgs
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if pkgs.len() < 2 {
            continue;
        }
        pkgs.sort_by_key(|p| (keeps_zero(p), p.as_str().to_string()));
        for pkg in pkgs.into_iter().skip(1) {
            if relocations.contains_key(&pkg) {
                continue;
            }
            let source = graph.package_table.get(&pkg).map(|p| &p.kind);
            let addr = published_original_id(&paths[&pkg], &env)?
                .unwrap_or_else(|| dummy_address(pkg, source));
            relocations.insert(pkg, addr);
        }
    }
    Ok(relocations)
}

/// `[published.<env>] original-id` from the package's `Published.toml`.
fn published_original_id(package_path: &Path, env: &str) -> Result<Option<AccountAddress>> {
    let path = package_path.join("Published.toml");
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(None);
    };
    let doc: toml::Value =
        toml::from_str(&text).with_context(|| format!("Parsing {}", path.display()))?;
    let Some(id) = doc
        .get("published")
        .and_then(|p| p.get(env))
        .and_then(|e| e.get("original-id"))
        .and_then(|v| v.as_str())
    else {
        return Ok(None);
    };
    let addr = AccountAddress::from_hex_literal(id)
        .with_context(|| format!("Invalid original-id '{id}' in {}", path.display()))?;
    Ok((addr != AccountAddress::ZERO).then_some(addr))
}

/// A stable address for an unpublished package that cannot stay at 0x0, derived from its name
/// and its source as the root declares it -- git url, rev and subdir, or the path relative to the
/// root -- so every machine derives the same one (move-package-alt's `dummy_addr`, widened to the
/// full address so it cannot land on a framework address).
fn dummy_address(pkg: PackageName, source: Option<&PM::DependencyKind>) -> AccountAddress {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"move-package-unpublished:");
    hasher.update(pkg.as_str().as_bytes());
    hasher.update(b":");
    match source {
        Some(PM::DependencyKind::Git(git)) => {
            hasher.update(git.git_url.as_str().as_bytes());
            hasher.update(b"@");
            hasher.update(git.git_rev.as_str().as_bytes());
            hasher.update(b"/");
            hasher.update(git.subdir.to_string_lossy().as_bytes());
        }
        Some(PM::DependencyKind::Local(path)) => hasher.update(path.to_string_lossy().as_bytes()),
        Some(PM::DependencyKind::OnChain(info)) => hasher.update(info.id.as_str().as_bytes()),
        None => {}
    }
    let mut bytes: [u8; AccountAddress::LENGTH] = hasher.finalize().into();
    bytes[0] |= 0x80;
    AccountAddress::new(bytes)
}

/// The `rename-from` of the dependency `dep_name`, if its declaration carries one.
fn package_rename(
    subst: Option<&PM::Substitution>,
    dep_name: &PackageName,
) -> Option<NamedAddress> {
    match subst?.get(dep_name)? {
        SubstOrRename::PackageRename(from) => Some(*from),
        _ => None,
    }
}

/// Whether a dependency key names the package it points at. Besides the manifest's
/// `package.name`, accept the name move-package-alt derives for a legacy manifest
/// (`derive_modern_name`): its single 0x0 / unassigned named address, else the one address its
/// modules are declared under. A legacy package named `Wormhole` whose modules are
/// `wormhole::*` is depended on as `wormhole = { ... }` by modern manifests, which
/// `sui move build` accepts. With `rename-from`, the renamed name is what must match.
fn dep_name_matches_manifest(
    dep_name: &PackageName,
    rename_from: Option<NamedAddress>,
    manifest: &SourceManifest,
    dep_path: &Path,
) -> bool {
    let wanted = rename_from.unwrap_or(*dep_name);
    if wanted == manifest.package.name {
        return true;
    }
    derived_package_names(manifest, dep_path).contains(wanted.as_str())
}

/// move-package-alt's modern name for a legacy manifest: its single 0x0 / unassigned named
/// address if there is exactly one, else the address names its modules are declared under
/// (a name only when that is unambiguous).
fn derived_package_names(manifest: &SourceManifest, dep_path: &Path) -> BTreeSet<String> {
    let zero: Vec<&PM::NamedAddress> = manifest
        .addresses
        .iter()
        .flatten()
        .filter(|(_, addr)| addr.is_none_or(|a| a == AccountAddress::ZERO))
        .map(|(name, _)| name)
        .collect();
    if zero.len() == 1 {
        return BTreeSet::from([zero[0].to_string()]);
    }
    let names = module_address_names(&dep_path.join(SourcePackageLayout::Sources.path()), 8);
    if names.len() == 1 {
        names
    } else {
        BTreeSet::new()
    }
}

/// The named addresses `module <addr>::<name>` declarations use under `dir`.
fn module_address_names(dir: &Path, depth: usize) -> BTreeSet<String> {
    module_decls(dir, depth)
        .into_iter()
        .map(|(addr, _)| addr)
        .collect()
}

/// Every `(address name, module name)` a `module <addr>::<name>` declaration under `dir` uses.
fn module_decls(dir: &Path, depth: usize) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    if depth == 0 {
        return out;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(module_decls(&path, depth - 1));
        } else if extension_equals(&path, "move") {
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            for line in text.lines() {
                let line = line.split("//").next().unwrap_or("").trim_start();
                let Some(rest) = line.strip_prefix("module ") else {
                    continue;
                };
                let Some((addr, module)) = rest.trim_start().split_once("::") else {
                    continue;
                };
                let addr = addr.trim();
                let module: String = module
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !addr.is_empty() && !addr.starts_with("0x") && !module.is_empty() {
                    out.insert((addr.to_string(), module));
                }
            }
        }
    }
    out
}

fn source_paths_for_config(package_path: &Path, config: &BuildConfig) -> Vec<PathBuf> {
    let mut places_to_look = Vec::new();
    let mut add_path = |layout_path: SourcePackageLayout| {
        let path = package_path.join(layout_path.path());
        if layout_path.is_optional() && !path.exists() {
            return;
        }
        places_to_look.push(path)
    };

    add_path(SourcePackageLayout::Sources);
    add_path(SourcePackageLayout::Scripts);

    if config.dev_mode {
        add_path(SourcePackageLayout::Examples);
        add_path(SourcePackageLayout::Tests);
    }

    places_to_look
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

fn package_digest_for_config(package_path: &Path, config: &BuildConfig) -> Result<PackageDigest> {
    let mut source_paths = source_paths_for_config(package_path, config);
    source_paths.push(package_path.join(SourcePackageLayout::Manifest.path()));
    compute_digest(&source_paths)
}
