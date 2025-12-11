CXX = g++
CXXFLAGS = -std=c++17 -Wall -pthread -O2
LDFLAGS = -lrt -pthread

TARGET = scheduler_service
SRCS = app.cpp logger.cpp shm_core.cpp scheduler.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf logs

.PHONY: all clean
