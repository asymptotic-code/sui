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
