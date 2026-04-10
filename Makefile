CXX      = g++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra \
           -mavx2 -mfma -ffast-math \
           -I.
LDFLAGS  =

TARGET  = tlumina
SRCS    = main.cpp core/model.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Build OK -> ./$(TARGET)  [or  ./$(TARGET) path/to/model.bin]"

clean:
	rm -f $(TARGET)

.PHONY: clean
