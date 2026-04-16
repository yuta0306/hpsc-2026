CXX = g++-15
CXXFLAGS = -fopenmp

00_hello: 00_hello.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
01_num_threads: 01_num_threads.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
02_loop: 02_loop.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
03_iterator: 03_iterator.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
04_scope: 04_scope.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
05_scope: 05_scope.cpp 05_bar.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@
06_sections: 06_sections.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
07_private: 07_private.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
08_tasks: 08_tasks.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
09_reduction: 09_reduction.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
10_atomic: 10_atomic.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
11_scan: 11_scan.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
12_pi: 12_pi.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
13_merge_sort: 13_merge_sort.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
14_bucket_sort: 14_bucket_sort.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
