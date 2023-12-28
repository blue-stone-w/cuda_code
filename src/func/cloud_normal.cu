#include "func/cloud_normal.h"

CloudNormal::CloudNormal( )
{}

CloudNormal ::~CloudNormal( )
{}

bool CloudNormal::getModelCurvatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int k)
{
  if (cloud->size( ) == 0)
  {
    return false;
  }

  pcl::gpu::NormalEstimation::PointCloud gpuCloud;

  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kdtree->setInputCloud(cloud);

  size_t cloud_size = cloud->points.size( );

  std::vector<float> dists;
  std::vector<std::vector<int>> neighbors_all;
  std::vector<int> sizes;
  neighbors_all.resize(cloud_size);
  sizes.resize(cloud_size);
#pragma omp parallel for
  for (int64_t i = 0; i < cloud_size; ++i)
  {
    kdtree->nearestKSearch(cloud->points[i], k, neighbors_all[i], dists);
    sizes[i] = (int)neighbors_all[i].size( );
  }
  int max_nn_size = *max_element(sizes.begin( ), sizes.end( ));
  std::vector<int> temp_neighbors_all(max_nn_size * cloud->size( ));
  pcl::gpu::PtrStep<int> ps(&temp_neighbors_all[0], max_nn_size * pcl::gpu::PtrStep<int>::elem_size);
  for (size_t i = 0; i < cloud->size( ); ++i)
    std::copy(neighbors_all[i].begin( ), neighbors_all[i].end( ), ps.ptr(i));

  pcl::gpu::NeighborIndices indices;
  gpuCloud.upload(cloud->points);
  indices.upload(temp_neighbors_all, sizes, max_nn_size);

  pcl::gpu::NormalEstimation::Normals normals;
  pcl::gpu::NormalEstimation::computeNormals(gpuCloud, indices, normals);
  pcl::gpu::NormalEstimation::flipNormalTowardsViewpoint(gpuCloud, 0.f, 0.f, 0.f, normals);

  std::vector<pcl::PointXYZ> downloaded;
  normals.download(downloaded);

  return true;
}