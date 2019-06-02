# we want to predict where to play a piece in connect four.
#I think we need to seperate into roles to make it easier.

# The dataset looks like this
#
#  
#   
#   
#
#

for i in range(0, 42):
    if i % 6 == 0:
        print("\n")
    print( " " ,i, " |  ", end = "")