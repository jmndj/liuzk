#------------------------------------------------------------------------------+

#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math
import os
import sys
import csv
import pandas as pd
#仅适用于立方晶系拉伸
###########################################
# -*- coding: utf-8 -*-


def matrix(N,a):
    b=c=a
    X1=1
    Y1=0
    Z1=0;
    X2=0
    Y2=1
    Z2=0
    X3=0
    Y3=0
    Z3=1
    x1=N[0][0]
    y1=N[0][1]
    z1=N[0][2]
    x2=N[1][0]
    y2=N[1][1]
    z2=N[1][2]
    x3=N[2][0]
    y3=N[2][1]
    z3=N[2][2]
    V=1/(X1*(Y2*Z3-Y3*Z2)-Y1*(X1*Z3-X3*Z2)+Z1*(X2*Y3-X3*Y2));
    a11=V*(x1*(Y2*Z3-Z2*Y3)+y1*(X3*Z2-X2*Z3)+z1*(X2*Y3-X3*Y2));
    a12=V*(x2*(Y2*Z3-Z2*Y3)+y2*(X3*Z2-X2*Z3)+z2*(X2*Y3-X3*Y2));
    a13=V*(x3*(Y2*Z3-Z2*Y3)+y3*(X3*Z2-X2*Z3)+z3*(X2*Y3-X3*Y2));
    a21=V*(x1*(Y3*Z1-Y1*Z3)+y1*(X1*Z3-X3*Z1)+z1*(X3*Y1-X1*Y3));
    a22=V*(x2*(Y3*Z1-Y1*Z3)+y2*(X1*Z3-X3*Z1)+z2*(X3*Y1-X1*Y3));
    a23=V*(x3*(Y3*Z1-Y1*Z3)+y3*(X1*Z3-X3*Z1)+z3*(X3*Y1-X1*Y3));
    a31=V*(x1*(Y1*Z2-Y2*Z1)+y1*(X2*Z1-X1*Z2)+z1*(X1*Y2-Y1*X2));
    a32=V*(x2*(Y1*Z2-Y2*Z1)+y2*(X2*Z1-X1*Z2)+z2*(X1*Y2-Y1*X2));
    a33=V*(x3*(Y1*Z2-Y2*Z1)+y3*(X2*Z1-X1*Z2)+z3*(X1*Y2-Y1*X2));


    B=1/(a11*(a22*a33-a23*a32)-a21*(a12*a33-a13*a32)+a31*(a12*a23-a13*a22));
    b11=B*(a22*a33-a23*a32);
    b12=B*(a13*a32-a12*a33);
    b13=B*(a12*a23-a13*a22);
    b21=B*(a23*a31-a21*a33);
    b22=B*(a11*a33-a13*a31);
    b23=B*(a21*a13-a11*a23);
    b31=B*(a21*a32-a22*a31);
    b32=B*(a12*a31-a11*a32);
    b33=B*(a11*a22-a21*a12);

    a=(b11*a,b21*a,b31*a)
    b=(b12*b,b22*b,b32*b)
    c=(b13*c,b23*c,b33*c)

    return(a,b,c)
# -*- coding: utf-8 -*-


#!/usr/bin/python

#仅在立方坐标系下使用,共有3个参数
import math
def Coordinate(Plane,Crystallographic,a):
    c=b=a
    o1= Plane[0]*a 
    o2= Plane[1]*b
    o3= Plane[2]*c
    o4=(o1*o1+o2*o2+o3*o3)**0.5
    x3=o1/o4
    y3=o2/o4
    z3=o3/o4   #晶面晶向指数
    o1= Crystallographic[0]*a
    o2= Crystallographic[1]*b
    o3= Crystallographic[2]*c
    o4=(o1*o1+o2*o2+o3*o3)**0.5
    x1=o1/o4
    y1=o2/o4
    z1=o3/o4   #剪切方向指数
    
    #晶面法向方向的theta与fai
    theta=math.acos(z3/((x3*x3+y3*y3+z3*z3)**0.5+float("1e-10")))/math.pi*180
    theta1=round(theta,3)
    if y3 > 0:
       fai=math.acos(x3/((x3*x3+y3*y3)**0.5+float("1e-10")))/math.pi*180
    else:
       fai=-math.acos(x3/((x3*x3+y3*y3)**0.5+float("1e-10")))/math.pi*180
    fai1=round(fai,3)
    #先绕z轴旋转fai角，再绕y轴旋转theta角
    x4=math.cos(theta)*math.cos(fai)*x1-math.cos(theta)*math.sin(fai)*y1+math.sin(theta)*z1
    y4=math.sin(fai)*x1+math.cos(fai)*y1
    z4=-math.sin(theta)*math.cos(fai)*x1+math.sin(theta)*math.sin(fai)*y1+math.cos(theta)*z1
    if y4 > 0:
        beta=math.acos(x4/((x4*x4+y4*y4)**0.5+float("1e-10")))/math.pi*180
    else:
        beta=-math.acos(x4/((x4*x4+y4*y4)**0.5+float("1e-10")))/math.pi*180
    beta1=round(beta,3)

    o1=y1*z3-y3*z1
    o2=-(x1*z3-x3*z1)
    o3=x1*y3-x3*y1
    o4=(o1*o1+o2*o2+o3*o3)**0.5
    x2=-o1/o4
    y2=-o2/o4
    z2=-o3/o4
    return(x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(fai1,theta1,beta1)

#--- COST FUNCTION ------------------------------------------------------------+
import Crystallographic as CC
import theory
from pymatgen.io.vasp.outputs import Vasprun
import xml.etree.ElementTree
import time
# function we are attempting to optimize (minimize)

def func1(x):
#仅适用于立方晶系拉伸
 
     #X=(1,0,0)
     #Y=(0,1,0)
     #Z=(0,0,1)
###########################################
     ##
    # N1=(-math.cos(x[0])*math.sin(x[1]),math.sin(x[0])*math.sin(x[1]),math.cos(x[1]))
    # N2=(math.cos(x[0])*math.cos(x[1])*math.cos(x[2])-math.sin(x[0])*math.sin(x[2]),-math.sin(x[0])*math.cos(x[1])*math.cos(x[2])-math.cos(x[0])*math.sin(x[2]),math.sin(x[1])*math.cos(x[2]))
    # N3=(CC.Coordinate(N1,N2,a))
 
    # result=theory.matrix(N3,a)
     
    # temp = sys.stdout
    # file = open('pos','w')
    # sys.stdout = file
    # print('diamond')
    # print(1.0)
    # print(result[0][0],result[0][1],result[0][2])
    # print(result[1][0],result[1][1],result[1][2])
    # print(result[2][0],result[2][1],result[2][2]) # 输出旋转后的基底 
    
    # sys.stdout.close()
    # temp = sys.stdout
    # file = open('angle','w')
    # sys.stdout = file
    # print(N3[3][0],N3[3][1],N3[3][2]) # 输出旋转angle
    # sys.stdout.close()
    # os.system('sh slurm.sh') 
    # flag = True
    # count=0 
    # while flag:
    #       try:
            # vasprun = Vasprun("vasprun.xml")#尝试打开vasprun.xml
    #         count += 1                      #若可以打开，证明一个数据算完，计数变量加一，同时删去vasprun.xml,等待下一个数据点算出
            # os.system('rm vasprun.xml')     
    #         if count > 2:
    #            break  # vasprun.xml is successfully closed
    #       except:
    #         time.sleep(10)
     if (os.path.exists('tmp.dat')):     
         out = open('tmp.csv','w',newline='')
         csv_writer = csv.writer(out,dialect='excel')
         f = open("tmp.dat","r",encoding='utf-8')    
         for line in f.readlines():
             mylist = line.split()
             csv_writer.writerow(mylist)                        #转化为csv文件
         f.close()
         out.close()
         os.system('rm tmp.dat')
         df = pd.read_csv('./tmp.csv', names=['shear', 'stress'])
         total= df['stress'].max()                            #找出最大值 
         result=-total
     else:
         result=0
     return result
  
#--- MAIN ---------------------------------------------------------------------+

class Particle:                     # 定义一个类
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=0          # best error individual
        self.err_i=0               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-0.8,0.8))     #random.uniform随机生成一个浮点数，
            self.position_i.append(x0[i])                    #

    # evaluate current fitness
    def evaluate(self,costFunc):                             #定义评估函数
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):                     #遍历每个自由度
            r1=random.random()                                #随机生成一个浮点数，范围在[0,1)之间
            r2=random.random()                                

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])   
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])              
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social  

    # update the particle position based off new velocity updates
    def update_position(self,bounds):                         
        for i in range(0,num_dimensions):                     #遍历每个粒子的自由度
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]   

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
    def directory(self,z):
        temp = sys.stdout
        file = open('dir.txt','w')
        sys.stdout = file
        print(z) # 输出旋转粒子标号
        sys.stdout.close()
        
        os.system('sh dir.sh')
        
    def sbatch(self,y):
        X=(1,0,0)
        Y=(0,1,0)
        Z=(0,0,1)
###########################################
        a=b=c=3.556
###########################################
        x=self.position_i
        N1=(-math.cos(x[0])*math.sin(x[1]),math.sin(x[0])*math.sin(x[1]),math.cos(x[1]))
        N2=(math.cos(x[0])*math.cos(x[1])*math.cos(x[2])-math.sin(x[0])*math.sin(x[2]),-math.sin(x[0])*math.cos(x[1])*math.cos(x[2])-math.cos(x[0])*math.sin(x[2]),math.sin(x[1])*math.cos(x[2]))
        N3=(CC.Coordinate(N1,N2,a))
 
        result=theory.matrix(N3,a)
     
        temp = sys.stdout
        file = open('pos','w')
        sys.stdout = file
        print('diamond')
        print(1.0)
        print(result[0][0],result[0][1],result[0][2])
        print(result[1][0],result[1][1],result[1][2])
        print(result[2][0],result[2][1],result[2][2]) # 输出旋转后的基底 
        
        sys.stdout.close()
        temp = sys.stdout
        file = open('sbatch.txt','w')
        sys.stdout = file
        print(y) # 输出旋转粒子标号
        sys.stdout.close()

        os.system('sh sbatch.sh')
        
    def readfile(self,w,):
        temp = sys.stdout
        file = open('readfile.txt','w')
        sys.stdout = file
        print(w) # 输出旋转粒子标号
        sys.stdout.close()

        os.system('sh readfile.sh')
class PSO():                                                  #定义一个类
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions                                #num_dimensions为全局变量

        num_dimensions=len(x0)                            
        err_best_g=0                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]                                              #构建粒子群
        for i in range(0,num_particles):                      #遍历所有的粒子               
            swarm.append(Particle(x0))
                        
        # 将所有粒子初始化                  
        swarm[0].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[1].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[2].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[3].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[4].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[5].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[6].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[7].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[8].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[9].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[10].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[11].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[12].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[13].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[14].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[15].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[16].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[17].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[18].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        swarm[19].position_i=[random.uniform(-3.14,3.14),random.uniform(0,3.14),random.uniform(-3.14,3.14)]
        
        for i in range(0,num_particles):
            swarm[i].directory(i)
        
        # begin optimization loop
        i=0
        while i < maxiter:
            #first=0
            #print i,err_best_g  
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                temp = sys.stdout
                file = open('num.txt','w')
                sys.stdout = file
                print(i) # 输出旋转粒子标号
                sys.stdout.close()
                os.system('sh num.sh')
                swarm[j].sbatch(j)
            #    first += 1
            #countp=0
            #nump=first
            flag=1
            while flag:                                              
                if (os.path.exists('watch.txt')):                    #通过watch.txt文件是否存在判断是否有粒子计算完毕
                    count=len(open(r"watch.txt",'rU').readlines())   #
                    #if count > countp:                     
                    #    if (count-countp) == 1:                           #如果只计算完一个
                    #       temp = sys.stdout
                    #       file = open('num.txt','w')
                    #       sys.stdout = file
                    #       print(nump) # 输出旋转粒子标号
                    #       sys.stdout.close()
                    #       swarm[nump].sbatch()    #提交下一个
                    #       nump += 1                                  #更新计数变量
                    #       countp=count                               #更新计数变量
                    #    else:
                    #        k=nump+count-countp                        #如果在10秒内算完不止一个，更新变量
                    #        for j in range(nump,k):
                    #            temp = sys.stdout
                    #            sys.stdout = file
                    #           print(j) # 输出旋转粒子标号
                    #            sys.stdout.close()
                    #            swarm[j].sbatch()
                    #            countp=count                                  #
                    #            nump=k
                    if count == num_particles:                         #
                        flag=0                                        #结束本次while循环                  
                        os.system('rm watch.txt')                     #本代全体粒子迭代结束                                
                time.sleep(10)
            total=0 
            for j in range(0,num_particles):                     #遍历每个粒子
                swarm[j].readflie(j)
                swarm[j].evaluate(costFunc)
                err_real_i=-swarm[j].err_best_i
                temp = sys.stdout
                file = open('{0}_best.txt'.format(j),'a+')
                sys.stdout = file
                print(i,swarm[j].pos_best_i,err_real_i) # 输出个体最优方向和最优值
                sys.stdout.close()
                err_real_i=-swarm[j].err_i
                temp = sys.stdout
                file = open('{0}.txt'.format(j),'a+')
                sys.stdout = file
                print(i,swarm[j].position_i,err_real_i) # 输出个体最优方向和最优值
                sys.stdout.close()
                total=total+err_real_i                  
                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == 0:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            average=total/num_particles
            temp = sys.stdout
            file = open('average.txt','a+')
            sys.stdout = file
            print(i,average) # 输出群体最优方向和最优值
            sys.stdout.close()
            err_best_real_g=-err_best_g
            temp = sys.stdout
            file = open('group_best.txt','a+')
            sys.stdout = file
            print(i,pos_best_g,err_best_real_g) # 输出群体最优方向和最优值
            sys.stdout.close()
            #将每次迭代的结果输出到指定文件里面

            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        err_best_g_real=-err_best_g
        temp = sys.stdout
        file = open('psoresult','w')
        sys.stdout = file
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g_real)
        sys.stdout.close()

        
        
        

if __name__ == "__PSO__":
    main()

#--- RUN ----------------------------------------------------------------------+
initial=[0,0,0]               # initial starting location [x1,x2...]
bounds=[(-6.28,6.28),(0,6.28),(-6.28,6.28)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1,initial,bounds,num_particles=20,maxiter=15)

#--- END ----------------------------------------------------------------------+
