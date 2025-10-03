class Cell{
    int hasBreeze;
    int hasGold;
    int hasStench;
    int hasWumpus;
    int hasPit;

    public Cell() {
        this.hasBreeze = -1;
        this.hasGold = -1;
        this.hasPit = -1;
        this.hasStench = -1;
        this.hasWumpus = -1;
    }

    public Cell(int hasBreeze, int hasStench, int hasWumpus, int hasGold, int hasPit){
        this.hasBreeze = hasBreeze;
        this.hasStench = hasStench;
        this.hasGold = hasGold;
        this.hasWumpus = hasWumpus;
        this.hasPit = hasPit;
    }

    public Cell(int all){
        this.hasBreeze = all;
        this.hasStench = all;
        this.hasGold = all;
        this.hasWumpus = all;
        this.hasPit = all;
    }

    @Override
    public String toString() {
        String output = "[";
        if(hasBreeze == 1){
            output += "B";
        }
        if(hasGold == 1){
            output += "G";
        }
        if(hasPit == 1){
            output += "P";
        }
        if(hasStench == 1){
            output += "S";
        }
        if(hasWumpus == 1){
            output += "G";
        }
        output += "]";
        return output;
    }
}

public class Wumpus{
    static Cell[][] makeBoard(int size, Object[][] locations){
        Cell[][] board = new Cell[size][size];
        for(int i = 0; i < board.length; i++){
            for (int j = 0; j < board.length; j++) {
                board[i][j] = new Cell(0);
            }
        }
        for(Object[] location: locations){
            int i = (int)location[0];
            int j = (int)location[1];
            char s = (char)location[2];
            if(s=='W'){
                if(i-1>-1){
                    board[i-1][j].hasStench = 1;
                }
                if(i+1<4){
                    board[i+1][j].hasStench = 1;
                }
                if(j-1>-1){
                    board[i][j-1].hasStench = 1;
                }
                if(j+1<4){
                    board[i][j+1].hasStench = 1;
                }
            }
            if(s=='P'){
                if(i-1>-1){
                    board[i-1][j].hasBreeze = 1;
                }
                if(i+1<4){
                    board[i+1][j].hasBreeze = 1;
                }
                if(j-1>-1){
                    board[i][j-1].hasBreeze = 1;
                }
                if(j+1<4){
                    board[i][j+1].hasBreeze = 1;
                }
            }
            if(s=='G'){
                board[i][j].hasGold = 1;
            }
        }
        return board;
    }

    static boolean recSolver(Cell[][] board, int[] loc, Cell[][] KB){
        int i = loc[0];
        int j = loc[1];
        System.out.print(String.format("-->(%d,%d)", i,j));
        if(board[i][j].hasGold == 1){
            System.out.println("  Found Gold");
            return true;
        }
        if(board[i][j].hasBreeze == 0){
            if(i+1<4){
                KB[i+1][j].hasPit = 0;
            }
            if(i-1>-1){
                KB[i-1][j].hasPit = 0;
            }
            if(j+1<4){
                KB[i][j+1].hasPit = 0;
            }
            if(j-1>-1){
                KB[i][j-1].hasPit = 0;
            }
        }
        if(board[i][j].hasStench == 0){
            if(i+1<4){
                KB[i+1][j].hasWumpus = 0;
            }
            if(i-1>-1){
                KB[i-1][j].hasWumpus = 0;
            }
            if(j+1<4){
                KB[i][j+1].hasWumpus = 0;
            }
            if(j-1>-1){
                KB[i][j-1].hasWumpus = 0;
            }
        }
        if(i-1>-1 && KB[i-1][j].hasWumpus == 0 && KB[i-1][j].hasPit == 0){
            if(recSolver(board, new int[]{i-1, j}, KB)){
                return true;
            }
        }
        if(j+1<4 && KB[i][j+1].hasWumpus == 0 && KB[i][j+1].hasPit == 0){
            if(recSolver(board, new int[]{i, j+1}, KB)){
                return true;
            }
        }
        if(j-1>-1 && KB[i][j-1].hasWumpus == 0 && KB[i][j-1].hasPit == 0){
            if(recSolver(board, new int[]{i, j-1}, KB)){
                return true;
            }
        }
        if(i+1<4 && KB[i+1][j].hasWumpus == 0 && KB[i+1][j].hasPit == 0){
            if(recSolver(board, new int[]{i+1, j}, KB)){
                return true;
            }
        } 
        return false;
    }

    static void solver(Cell[][] board){
        Cell[][] KB = new Cell[board.length][board.length];
        for (int i = 0; i < KB.length; i++) {
            for (int j = 0; j < KB.length; j++) {
                KB[i][j] = new Cell();
            }
        }
        recSolver(board, new int[]{3,0}, KB);
    }

    public static void main(String[] args) {
        Object[][] locations = {{0,2,'P'},{1,0,'W'},{2,3,'P'},{3,2,'P'},{1,1,'G'}};
        Cell[][] board = makeBoard(4, locations);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
        solver(board);
    }
}