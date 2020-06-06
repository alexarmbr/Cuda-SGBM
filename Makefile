
helper: lib/sgbm_helper.cpp
	gcc -lstdc++ -shared -o lib/sgbm_helper.so lib/sgbm_helper.cpp

tests: lib/tests.cpp
	g++ lib/tests.cpp -o lib/tests.so
	