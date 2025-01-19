# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -std=c++11 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

# Output binary
TARGET = sobel_filter.out

# Source files
SRC = CUDA.cu

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)
