
helper: lib/sgbm_helper.cpp
	gcc -lstdc++ -shared -o lib/sgbm_helper.so lib/sgbm_helper.cpp

tests: lib/tests.cpp
	g++ lib/tests.cpp -o lib/tests.so


sgbm: sgbm.cpp
	g++ sgbm.cpp -o lib/sgbm.so -std=c++11 -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core