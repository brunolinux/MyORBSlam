file(GLOB SOURCE_FILES "*.cpp")
add_executable(catch2_test ${SOURCE_FILES})

target_link_libraries(catch2_test PRIVATE Catch2::Catch2)
target_link_libraries(catch2_test PRIVATE Extractor)

# old method
#include(ParseAndAddCatchTests)
#ParseAndAddCatchTests(catch2_test)

include(CTest)
include(Catch)

catch_discover_tests(catch2_test)
