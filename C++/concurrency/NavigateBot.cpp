#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <chrono>

using namespace std;

const int ROWS = 10;
const int COLS = 10;

enum class Cell {
    EMPTY,
    WALL,
    AGENT,
    DESTINATION,
    VISITED
};

struct Position {
    int row;
    int col;
};

class Maze {
private:
    vector<vector<Cell>> grid;
    Position source;
    Position destination;

public:
    Maze() {
        grid.resize(ROWS, vector<Cell>(COLS, Cell::EMPTY));
        generateMaze();
        placeAgentAndDestination();
    }

    void generateMaze() {
        srand(time(nullptr));
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                grid[i][j] = (rand() % 5 == 0) ? Cell::WALL : Cell::EMPTY;
            }
        }
    }

    void placeAgentAndDestination() {
        source.row = rand() % ROWS;
        source.col = rand() % COLS;
        grid[source.row][source.col] = Cell::AGENT;

        do {
            destination.row = rand() % ROWS;
            destination.col = rand() % COLS;
        } while (destination.row == source.row && destination.col == source.col);

        grid[destination.row][destination.col] = Cell::DESTINATION;
    }

    void displayMaze() {
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                switch(grid[i][j]) {
                    case Cell::EMPTY:
                        cout << " ";
                        break;
                    case Cell::WALL:
                        cout << "#";
                        break;
                    case Cell::AGENT:
                        cout << "A";
                        break;
                    case Cell::DESTINATION:
                        cout << "D";
                        break;
                    case Cell::VISITED:
                        cout << ".";
                        break;
                }
                cout << " ";
            }
            cout << endl;
        }
    }

    bool isInsideMaze(int row, int col) {
        return row >= 0 && row < ROWS && col >= 0 && col < COLS;
    }

    bool isDestination(int row, int col) {
        return row == destination.row && col == destination.col;
    }

    bool isTraversable(int row, int col) {
        return isInsideMaze(row, col) && grid[row][col] != Cell::WALL && grid[row][col] != Cell::VISITED;
    }

    void bfs() {
        queue<Position> q;
        q.push(source);

        while (!q.empty()) {
            Position current = q.front();
            q.pop();

            if (isDestination(current.row, current.col)) {
                grid[current.row][current.col] = Cell::AGENT;
                return;
            }

            if (isTraversable(current.row, current.col)) {
                grid[current.row][current.col] = Cell::VISITED;
                displayMaze();  // Display the maze after each move
                std::this_thread::sleep_for(std::chrono::seconds(1));  // Introduce 1 second delay

                if (isInsideMaze(current.row - 1, current.col)) q.push({current.row - 1, current.col});
                if (isInsideMaze(current.row + 1, current.col)) q.push({current.row + 1, current.col});
                if (isInsideMaze(current.row, current.col - 1)) q.push({current.row, current.col - 1});
                if (isInsideMaze(current.row, current.col + 1)) q.push({current.row, current.col + 1});
            }
        }
    }

    void traverseMaze() {
        bfs();
        displayMaze();
    }
};

int main() {
    Maze maze;
    maze.traverseMaze();

    return 0;
}
