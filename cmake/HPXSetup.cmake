if(WITH_CUSTOM_HPX)
    # Find a system or user-provided HPX installation
    find_package(HPX 1.11.0 REQUIRED)
else()
    # Fetch and build HPX as part of this project
    include(FetchContent)
    FetchContent_Declare(HPX
        URL https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/v1.11.0.tar.gz
        TLS_VERIFY TRUE
    )

    FetchContent_GetProperties(HPX)
    if(NOT HPX_POPULATED)
        # Set as CACHE variables so they can be easily overridden by the user
        set(HPX_WITH_CXX_STANDARD 20 CACHE STRING "C++ standard to use for HPX")
        set(HPX_WITH_MALLOC "system" CACHE STRING "The memory allocator to be used by HPX")
        set(HPX_WITH_FETCH_ASIO ON CACHE BOOL "")
        set(HPX_WITH_FETCH_BOOST ON CACHE BOOL "")
        set(HPX_WITH_FETCH_HWLOC ON CACHE BOOL "")

        # Disable parts we don't need for a dependency build
        set(HPX_WITH_EXAMPLES OFF CACHE BOOL "")
        set(HPX_WITH_TESTS OFF CACHE BOOL "")

        # Disabling GPU support
        set(HPX_WITH_CUDA OFF CACHE BOOL "")
        set(HPX_WITH_HIP OFF CACHE BOOL "")
        set(HPX_WITH_SYCL OFF CACHE BOOL "")

        # Other options
        set(HPX_WITH_DYNAMIC_HPX_MAIN ON CACHE BOOL "")
        set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries for dependencies")

        FetchContent_MakeAvailable(HPX)
    endif()
endif()
