// Special "spec functions"

// 1. for validator_set

// 1.a. find_validator.  The reatment here of an Option return is modelled on `vec_map::get_idx_opt`

procedure {:inline 1} $3_validator_set_find_validator_sf(v: Vec($3_validator_Validator), a: int)
  returns ($ret0: $1_option_Option'u64') {
  var r: int;
  r := find_validator_sf(v, a);
  if (r >= 0) {
     $ret0 := $1_option_Option'u64'(MakeVec1(r));
  } else {
     $ret0 := $1_option_Option'u64'(EmptyVec());
  }
}

function find_validator_sf(v: Vec($3_validator_Validator), a: int): int;

// callable from Move
procedure {:inline 1} $3_validator_set_exists_validator_in_range_sf(v: Vec($3_validator_Validator), lb: int, hb: int, a: int)
  returns (e: bool) {
  e := exists_validator_in_range_sf(v, lb, hb, a);
}

// callable from Boogie
function {:inline} exists_validator_in_range_sf(v: Vec($3_validator_Validator), lb: int, hb: int, a: int): bool {
 (exists i: int :: $IsValid'u64'(i) && InRangeVec(v, i) && lb <= i && i < hb && validator_sui_address(ReadVec(v,i)) == a)}

function {:inline} validator_sui_address(v: $3_validator_Validator): int {
  v->$metadata->$sui_address
}

// find_validator: if a validator with the given sui_address exists, the smallest index of a location is returned, otherwise -1.  In move, this is turned into an Option<u64>
axiom (forall v: Vec($3_validator_Validator), a: int ::
  { find_validator_sf(v, a) }
  (var r :=  find_validator_sf(v, a);
  if (exists_validator_in_range_sf(v, 0, LenVec(v), a))
  then $IsValid'u64'(r) && InRangeVec(v, r) && validator_sui_address( ReadVec(v, r) ) == a &&
       (! exists_validator_in_range_sf(v, 0, r, a))
  else r == -1));

// 1.b. for sum_voting_power_by_address

// public native fun all_addresses_exist_sf(vs: &vector<Validator>, addresses: &vector<address>): bool;
procedure {:inline 2} $3_validator_set_all_addresses_exist_sf(
           vs: Vec($3_validator_Validator),
           addresses: Vec(int)) returns (e: bool) {
    e := (forall i: int :: InRangeVec(addresses, i) ==> exists_validator_in_range_sf(vs, 0, LenVec(vs), ReadVec(addresses, i)));
    }

// public native fun sum_voting_power_by_addresses_range_sf(vs: &vector<Validator>, addresses: &vector<address>,
//     from: u64, to: u64): Integer;
// ... callable from move
procedure {:inline 1} $3_validator_set_sum_voting_power_by_addresses_range_sf(
   vs: Vec($3_validator_Validator),
   addresses: Vec(int),
   from: int,
   to: int) returns (sum: int) {
   sum := sum_voting_power_by_addresses_range_sf(vs, addresses, from, to);
   }
// ... and from Boogie
function sum_voting_power_by_addresses_range_sf(
   vs: Vec($3_validator_Validator),
   addresses: Vec(int),
   from: int,
   to: int): int; // defined by axioms

// the sum over an empty range is zero
axiom (forall vs: Vec($3_validator_Validator), addresses: Vec(int), from: int, to: int ::
   { sum_voting_power_by_addresses_range_sf(vs, addresses, from, to)}
   (from >= to ==> sum_voting_power_by_addresses_range_sf(vs, addresses, from, to) == 0));

// the sum of a range can be split in two
axiom (forall vs: Vec($3_validator_Validator), addresses: Vec(int), a: int, b: int, c: int, d: int ::
  { sum_voting_power_by_addresses_range_sf(vs, addresses, a, b), sum_voting_power_by_addresses_range_sf(vs, addresses, c, d) }
  0 <= a && a <= b && b == c && c <= d && d <= LenVec(addresses) ==>
    sum_voting_power_by_addresses_range_sf(vs, addresses, a, b)
    + sum_voting_power_by_addresses_range_sf(vs, addresses, c, d)
    == sum_voting_power_by_addresses_range_sf(vs, addresses, a, d)) ;

// the sum over a singleton range is the vector element there
axiom (forall vs: Vec($3_validator_Validator), addresses: Vec(int), a: int, x: int, y: int ::
   { sum_voting_power_by_addresses_range_sf(vs, addresses, x, y),
     addresses->v[a] }
    // in a proof involving sum_voting_power_by_addresses_range_sf(vs, addresses,...) and v[a]
   0 <= a && a + 1 <= LenVec(addresses)  ==> sum_voting_power_by_addresses_range_sf(vs, addresses, a, a+1) ==
   ReadVec(vs, find_validator_sf(vs, ReadVec(addresses, a)))->$voting_power);

// for vectors of u64, nested ranges have sums bounded by the larger
axiom (forall vs: Vec($3_validator_Validator), addresses: Vec(int), a: int, b: int, c: int, d: int ::
  { sum_voting_power_by_addresses_range_sf(vs, addresses, a, d), sum_voting_power_by_addresses_range_sf(vs, addresses, b, c) }
  $IsValid'vec'$3_validator_Validator''(vs) && 0 <= a && a <= b && b <= c && c <= d && d <= LenVec(addresses)  ==> sum_voting_power_by_addresses_range_sf(vs, addresses, b, c) <= sum_voting_power_by_addresses_range_sf(vs, addresses, a, d)) ;
