import numpy as np
class SudokuSolver:
    def __init__(self, board):
        self.board = board

    def solve(self):
        if self._solve_sudoku():
            return self._print_board()
        else:
            print("No solution exists for the given Sudoku board.")

    def _solve_sudoku(self):
        empty_cell = self._find_empty_cell()
        if not empty_cell:
            return True
        else:
            row, col = empty_cell

        for num in range(1, 10):
            if self._is_valid_move(row, col, num):
                self.board[row][col] = num

                if self._solve_sudoku():
                    return True
                else:
                    self.board[row][col] = 0

        return False

    def _is_valid_move(self, row, col, num):
        return (
            self._is_valid_row(row, num)
            and self._is_valid_column(col, num)
            and self._is_valid_box(row - row % 3, col - col % 3, num)
        )

    def _is_valid_row(self, row, num):
        for col in range(9):
            if self.board[row][col] == num:
                return False
        return True

    def _is_valid_column(self, col, num):
        for row in range(9):
            if self.board[row][col] == num:
                return False
        return True

    def _is_valid_box(self, start_row, start_col, num):
        for row in range(3):
            for col in range(3):
                if self.board[row + start_row][col + start_col] == num:
                    return False
        return True

    def _find_empty_cell(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:
                    return (row, col)
        return None

    def _print_board(self):
        rows=[]
        for row in self.board:
            rows.extend(row)
        #print(rows)
        return rows
board=[
      [5,0,0,0,0,0,0,9,0],
      [0,1,4,3,0,0,0,2,0],
      [0,0,0,0,0,0,3,0,7],
      [0,0,7,5,1,3,0,4,0],
      [0,0,1,0,4,0,7,0,0],
      [0,9,0,2,8,7,1,0,0],
      [6,0,2,0,0,0,0,0,0],
      [0,7,0,0,0,8,2,1,0],
      [0,8,0,0,0,0,0,0,6]]
solve=SudokuSolver(board)
a=solve.solve()
#print(a)
