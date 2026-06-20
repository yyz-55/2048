CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS  := -pthread

TARGET   := 2048
SRC      := main.cpp

.PHONY: all clean run bench

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) --play 1

bench: $(TARGET)
	./$(TARGET) --bench 10

clean:
	rm -f $(TARGET)
