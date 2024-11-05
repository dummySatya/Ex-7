# Directories for libraries and headers
INCLUDE_DIR = /home/satya-tt0492/onnxruntime-linux-x64-1.20.0/include
LIB_DIR = /home/satya-tt0492/onnxruntime-linux-x64-1.20.0/lib
SAMPLERATE_INCLUDE = /usr/include
SNDFILE_INCLUDE = /usr/include
LIBSAMPLERATE_DIR = /usr/lib/x86_64-linux-gnu
LIBSNDFILE_DIR = /usr/lib/x86_64-linux-gnu

# Compiler and flags
CXX = g++
CXXFLAGS = -I $(INCLUDE_DIR) -I $(SAMPLERATE_INCLUDE) -I $(SNDFILE_INCLUDE)
LDFLAGS = -L $(LIB_DIR) -L $(LIBSAMPLERATE_DIR) -L $(LIBSNDFILE_DIR)

# Libraries
LIBS = -lonnxruntime -lsndfile -lsamplerate

# Target executable
TARGET = my_program

# Source files
SOURCES = cppcode.cpp

# Build the target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LIBS) -o $(TARGET)
