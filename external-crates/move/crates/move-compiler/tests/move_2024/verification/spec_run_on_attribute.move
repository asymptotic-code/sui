module a::m {

    // Test run_on attribute with "local" value
    #[spec(prove, run_on=b"local")]
    fun test_spec_with_run_on_local() {}

    // Test run_on attribute with "remote" value
    #[spec(prove, run_on=b"remote")]
    fun test_spec_with_run_on_remote() {}

    // Test run_on with other spec attributes
    #[spec(prove, focus, run_on=b"local", timeout=60)]
    fun test_spec_with_multiple_attributes() {}

    // Test bare spec without run_on
    #[spec(prove)]
    fun test_spec_without_run_on() {}
}
