cmake_minimum_required(VERSION 3.16)
project(CMFOptimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Kütüphaneleri bul
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED)
find_package(cereal REQUIRED)

# Başlık-sadece kütüphaneleri ekle (proje dizinine kopyalandığı varsayılarak)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include # bs_thread_pool.hpp, csv.h, cxxopts.hpp vb. buraya konulabilir
)

add_executable(optimizer optimizer.cpp)

# Kütüphaneleri bağla
target_link_libraries(optimizer PRIVATE 
    Eigen3::Eigen
    HDF5::HDF5
    cereal::cereal
)

# Gerekirse HighFive'ı manuel olarak ekle
# (vcpkg veya benzeri bir araçla yüklendiyse bu daha kolay olur)
# find_package(HighFive REQUIRED)
# target_link_libraries(optimizer PRIVATE HighFive::HighFive)
# Şu an için HighFive'ın başlıklarını include path'e eklemek yeterli olabilir
# eğer HDF5 doğru şekilde bulunursa.

target_compile_features(optimizer PRIVATE cxx_std_17)
set_target_properties(optimizer PROPERTIES
    CXX_EXTENSIONS OFF)