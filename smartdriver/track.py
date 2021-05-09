import arcade
from math import sqrt
from smartdriver.constants import *

class Track():
    
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints
        self.checkpoints.append(checkpoints[0])

        self.next_checkpoint = 1
    
    def draw_track(self):
        if self.next_checkpoint >= len(self.checkpoints):
            arcade.draw_polygon_outline(self.checkpoints, TRACK_COLOR_PASSED, line_width=3)
        else:
            arcade.draw_polygon_outline(self.checkpoints[:-1], TRACK_COLOR_FUTURE, line_width=3)
            A, B = self.checkpoints[self.next_checkpoint-1:self.next_checkpoint+1]
            arcade.draw_line(A[0],A[1],B[0],B[1], TRACK_COLOR_CURRENT, line_width=3)    
            if self.next_checkpoint > 1:
                for i in range(self.next_checkpoint-1):
                    A, B = self.checkpoints[i], self.checkpoints[i+1]
                    arcade.draw_line(A[0],A[1],B[0],B[1], TRACK_COLOR_PASSED, line_width=3)    
                #arcade.draw_lines(self.checkpoints[:self.next_checkpoint-1], TRACK_COLOR_PASSED, line_width=3)        

    def change_current_segment(self, A, B, C):
        # segment AB smo prečkali, gremo na BC, ustrezno se spremenijo barve
        arcade.draw_line(A[0],B[0],A[1],B[1], TRACK_COLOR_PASSED, line_width=3)
        arcade.draw_line(B[0],C[0],B[1],C[1], TRACK_COLOR_CURRENT, line_width=3)

    def distance_to_track(self, position, last_checkpoint):
        A, B = self.checkpoints[last_checkpoint], self.checkpoints[last_checkpoint+1]
        return minDistance(A, B, position)

# Function to return the minimum distance
# between a line segment AB and a point E
def minDistance(A, B, E) :

    # vector AB
    AB = [None, None];
    AB[0] = B[0] - A[0];
    AB[1] = B[1] - A[1];

    # vector BP
    BE = [None, None];
    BE[0] = E[0] - B[0];
    BE[1] = E[1] - B[1];

    # vector AP
    AE = [None, None];
    AE[0] = E[0] - A[0];
    AE[1] = E[1] - A[1];

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1];
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1];

    # Minimum distance from
    # point E to the line segment
    reqAns = 0;

    # Case 1
    if (AB_BE > 0) :

        # Finding the magnitude
        y = E[1] - B[1];
        x = E[0] - B[0];
        reqAns = sqrt(x * x + y * y);

    # Case 2
    elif (AB_AE < 0) :
        y = E[1] - A[1];
        x = E[0] - A[0];
        reqAns = sqrt(x * x + y * y);

    # Case 3
    else:

        # Finding the perpendicular distance
        x1 = AB[0];
        y1 = AB[1];
        x2 = AE[0];
        y2 = AE[1];
        mod = sqrt(x1 * x1 + y1 * y1);
        reqAns = abs(x1 * y2 - y1 * x2) / mod;
    
    return reqAns;