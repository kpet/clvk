struct ParamStruct {
    uint a;
};

kernel void test_simple(global uint* out)
{
    struct ParamStruct a;
    size_t gid = get_global_id(0);
    out[gid] = gid;
}
