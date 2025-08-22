if(WITH_CUSTOM_HPX)
    find_package(HPX REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(HPX
        URL https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/v1.11.0.tar.gz
        TLS_VERIFY true
    )
    FetchContent_GetProperties(HPX)

    if(NOT HPX_POPULATED)
        set(HPX_WITH_CXX_STANDARD 20  CACHE INTERNAL "")
        set(HPX_WITH_FETCH_ASIO ON  CACHE INTERNAL "")
        set(HPX_WITH_MALLOC "system"  CACHE INTERNAL "")
        set(HPX_WITH_FETCH_BOOST ON  CACHE INTERNAL "")
        set(HPX_WITH_FETCH_HWLOC ON  CACHE INTERNAL "")
        set(HPX_WITH_EXAMPLES OFF  CACHE INTERNAL "")
        set(HPX_WITH_TESTS OFF  CACHE INTERNAL "")
        FetchContent_MakeAvailable(HPX)
    endif()
endif()
