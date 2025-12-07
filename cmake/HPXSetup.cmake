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

        find_package(Hwloc QUIET)

        # Determining if Hwloc needs to be fetched or not
        if(Hwloc_FOUND)
            message(STATUS "System hwloc found (v${Hwloc_VERSION}). Disabling internal HPX fetch.")
            set(HPX_WITH_FETCH_HWLOC OFF CACHE BOOL "Use system hwloc")
            set(HWLOC_ROOT "${Hwloc_INCLUDE_DIRS}" CACHE PATH "" FORCE)
        else()
            message(STATUS "System hwloc not found. Enabling internal HPX fetch.")
            set(HPX_WITH_FETCH_HWLOC ON CACHE BOOL "Fetch hwloc internally")
        endif()

        # Disable parts we don't need for a dependency build
        set(HPX_WITH_EXAMPLES OFF CACHE BOOL "")
        set(HPX_WITH_TESTS OFF CACHE BOOL "")

        # Other options
        set(HPX_WITH_DYNAMIC_HPX_MAIN ON CACHE BOOL "")
        set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries for dependencies")

        FetchContent_MakeAvailable(HPX)
    endif()
endif()
