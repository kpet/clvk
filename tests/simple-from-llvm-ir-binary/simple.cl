kernel void test_simple(global uint* out)
{
    size_t gid = get_global_id(0);
    out[gid] = gid;
}
