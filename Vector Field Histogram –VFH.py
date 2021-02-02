import numpy as np                   #Dhlwsh vivliothikwn
import matplotlib.pyplot as plt

def plot(k):
    plt.figure(k)
    plt.draw()
    plt.pause(0.0000000000001)

def Goal_Found():
    global robot, goal, mr, mc
    mpoints = max(abs(int(goal[0] - robot[0])), abs(int(goal[1]-robot[1])))
    mr = np.round(np.linspace(int(robot[0]), int(goal[0]), int(mpoints+1)))
    mc = np.round(np.linspace(int(robot[1]), int(goal[1]), int(mpoints+1)))
    for i,row_element in enumerate(space):
        for j,column_element in enumerate(space[i]):
            if i < len(mr)-1:
                k=int(mr[i])
                l=int(mc[i])
                space[k,l] = 3
    plot(1)
    titlos = plt.title('Goal Found')
    DispGrid(titlos)
    newpos = followmline()
    return True,newpos

def followmline():
    global mr, mc
    for element_mr, element_mc in zip(mr,mc):
        newpos = int(element_mr), int(element_mc)
        robot = newpos
        plot(1)
        plt.text(robot[1]-1,robot[0]+1,'R')
    return newpos


def DispGrid(titlos):
    colorarr = ['white', 'green', 'purple', 'yellow', 'red']
    global space,robot
    r,c = space.shape
    plot(1)
    for i in range(0,r):
        for j in range(0,c):
            color = colorarr[int(space[i,j])]
            plt.fill([j-1,j,j,j-1],[i,i,i+1,i+1],color)
    plt.text(robot[1]-1,robot[0]+1,'R')


def Obstacle(linex,liney):
    global space, robot, d, GoalFound
    obstacle_pose = []
    count = 0
    for element_linex, element_liney in zip(linex,liney):
        if np.round(element_linex) < d and np.round(element_liney) < d and np.round(element_linex) >= 0 and np.round(element_liney) >= 0:
            if space[int(np.round(element_liney)),int(np.round(element_linex))] == 2:
                obstacle_pose.append([element_linex,element_liney])
                count = 1
            if space[int(np.round(element_liney)),int(np.round(element_linex))] != 2 and count == 0:
                if np.round(element_linex) == goal[1] and np.round(element_liney) == goal[0]:
                    if not GoalFound:
                        GoalFound, robot = Goal_Found()
    if len(obstacle_pose) > 0:
        if linex[-1] <= robot[1]:
            obstacle_pose = max(obstacle_pose)
        if linex[-1] > robot[1]:
            obstacle_pose = min(obstacle_pose)
        obstacle_distance_x,obstacle_distance_y = obstacle_pose
        obstacle_distance = np.sqrt((obstacle_distance_x - robot[1])**2+(obstacle_distance_y-robot[0])**2)
        return obstacle_distance
    else:
        return 0


d= 30
space = np.zeros((d,d))
start = [27,3]
goal = [3,27]

space[start[0],start[1]] = 1
space[goal[0],goal[1]]=4
robot = start

ObstRows = [29, 29, 29, 28, 28, 28, 27, 27, 27,
            29, 29, 29, 29, 28, 28, 28, 28, 27, 27, 27, 27]
ObstCols = [10, 11, 12, 10, 11, 12, 10, 11, 12,
            22, 23, 24, 25, 22, 23, 24, 25, 22, 23, 24, 25]
ObstRows1 = [23, 23, 23, 22, 22, 22, 21, 21, 21, 23, 23, 23, 22, 22, 22, 21, 21, 21,
             23, 23, 23, 23, 22, 22, 22, 22, 21, 21, 21, 21]
ObstCols1 = [4, 5, 6, 4, 5, 6, 4, 5, 6, 15, 16, 17, 15, 16, 17, 15, 16, 17,
             19, 20, 21, 22, 19, 20, 21, 22, 19, 20, 21, 22]
ObstRows2 = [16, 16, 16, 15, 15, 15, 14, 14, 14, 16, 16, 16, 15, 15, 15, 14, 14, 14,
             16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14]
ObstCols2 = [3, 4, 5, 3, 4, 5, 3, 4, 5, 12, 13, 14, 12, 13, 14, 12, 13, 14,
             25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 27, 28]
ObstRows3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7,
             7, 7, 7, 7, 7, 7, 8, 8, 8, 7, 7, 7]
ObstCols3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 22, 23, 24]

    

for i in range (0, len(space)):
    for j in range (0, len(space[i])):
        space[0,j] = 2
        space[-1,j] = 2
        space[i, 1] = 2
        space[i, -1] = 2
        if i < len(ObstRows):
            space[ObstRows[i], ObstCols[i]] = 2
        if i < len(ObstRows1):
            space[ObstRows1[i], ObstCols1[i]] = 2
        if i < len(ObstRows2):
            space[ObstRows2[i], ObstCols2[i]] = 2
        if i < len(ObstRows3):
            space[ObstRows3[i], ObstCols3[i]] = 2

plot(1)
plt.axis([0,d,d,0])
plt.xticks(np.arange(0,d+1,1))
plt.yticks(np.arange(0,d+1,1))
titlos = plt.title('Searching for Obstacles')
plt.grid(True,color='black',linewidth = 0.5)

Openspace = []


DispGrid(titlos)
plot(1)
GoalFound = False
GoalSeek = True

while not GoalFound:
    if GoalSeek:
        x = np.linspace(0, +10, 20)
        y = 0
        for theta in range(-180, 190, 10):
            rad = np.deg2rad(theta)
            x_rotation = np.cos(rad)*x - np.sin(rad)*y
            y_rotation = np.sin(rad)*x + np.cos(rad)*y
            plot(1)
            plt.title('Searching for Obstacles')
            line, = plt.plot(robot[1] - 0.5 + x_rotation, robot[0] + 0.5 + y_rotation, '-r')
            posx = robot[1] - 0.5 + x_rotation
            posy = robot[0] + 0.5 + y_rotation
            data = (Obstacle(posx,posy))
            if data == 0:
                Openspace.append([theta,[posx,posy]])
            plot(2)
            plt.title('Vector Field Histogram')
            plt.bar(theta, data, width = 0.8, color=(1.0, 0.0, 0.0, 1.0))
            plt.xticks(np.arange(-180,181,90))
            line.remove()
            if np.array_equal(robot,goal):
                GoalFound = True
                print(GoalFound)
                break
        distance = 100
        for elements_Openspace in Openspace:
            theta, closer_position = elements_Openspace
            closer_position_x, closer_position_y = closer_position
            distance_of_goal = np.sqrt((closer_position_x[-1] - goal[1])**2 + (closer_position_y[-1] - goal[0])**2)
            if distance_of_goal < distance:
                closer_x = closer_position_x[-1]
                closer_y = closer_position_y[-1]
                distance = distance_of_goal
        mpoints = max(abs(int(closer_y) - robot[0]), abs(int(closer_x)-robot[1]))
        mr = np.round(np.linspace(int(robot[0]), int(np.round(closer_y)), int(mpoints+1)))
        mc = np.round(np.linspace(int(robot[1]), int(np.round(closer_x)), int(mpoints+1)))
        if mpoints < 2:
            distance = 0
            for elements_Openspace in Openspace:
                theta, further_position = elements_Openspace
                further_position_x, further_position_y = closer_position
                distance_of_goal = np.sqrt((further_position_x[5] - goal[1])**2 + (further_position_y[5] - goal[0])**2)
                if distance_of_goal > distance:
                    further_x = further_position_x[5]
                    further_y = further_position_y[5]
                    distance = distance_of_goal
            mpoints = max(abs(int(np.round(further_y)) - robot[0]), abs(int(np.round(further_x))-robot[1]))
            mr = np.round(np.linspace(int(robot[0]), int(np.round(further_y)), int(mpoints+1)))
            mc = np.round(np.linspace(int(robot[1]), int(np.round(further_x)), int(mpoints+1)))  
        for i,row_element in enumerate(space):
            for j,column_element in enumerate(space[i]):
                if i < len(mr):
                    k=int(mr[i])
                    l=int(mc[i])
                    space[k,l] = 3
        if not GoalFound:
            DispGrid(titlos)
            plot(1)
            titlos = plt.title('Follow MLine')
            robot = followmline()
        plot(2)
        plt.clf()
                 
            
                   
            
            
