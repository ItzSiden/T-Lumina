CXX      = g++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra
TARGET   = tlumina
SRCS     = main.cpp core/model.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
	@echo "✅ Build done: ./$(TARGET)"

# তোমার নিজের model চালাতে
run: $(TARGET)
	./$(TARGET) tlumina_model.bin config.json

clean:
	rm -f $(TARGET)