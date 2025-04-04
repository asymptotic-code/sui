module 0x1::M {
    #[spec_only]
    public struct Foo {}

    // This should cause an unbound type error in non-verify mode
    // as the Foo struct declaration was filtered out
    public fun foo(): Foo {
        Foo {}
    }

    #[spec_only]
    public fun bar() {
    }

    // This should cause an unbound function error in non-verify mode
    // as `bar` was filtered out
    public fun baz() {
        bar()
    }
}
