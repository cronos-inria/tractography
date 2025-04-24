inline float3 to_voxel(float4 affine[4], float4 point) {
	float3 voxel;
    for (size_t i = 0; i < 3; i++) {
        voxel[i] = dot(affine[i], point);
    }
	return voxel;
}

inline bool in_image(float3 voxel, uint nx, uint ny, uint nz) {
	return !(voxel.x < 0 || voxel.x >= nx || voxel.y < 0 || voxel.y >= ny || voxel.z < 0 || voxel.z >= nz);
}

inline uint3 to_index(float3 voxel) {
	uint3 index = {(uint) rint(voxel.x), (uint) rint(voxel.y), (uint) rint(voxel.z)};
	return index;
}
