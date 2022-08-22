import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self,alpha=0.001,lambda_=5):
        self.alpha=alpha
        self.lambda_=lambda_
    def cost_func(self,w,b):
        return (sum((np.dot(x,np.reshape(w,(-1,1)))+b \
                  -np.reshape(y,(len(y),-1)))**2)/(2*len(x)))[0]

    def derivative(self,w,b,wth=None,by="w"):
        cem2= np.dot(x,np.reshape(w,(-1,1)))+b - np.reshape(y,(-1,1))
        if by=='w':
            cem2*=np.array(x)[:,wth].reshape((-1,1))
        return (sum(cem2)/len(x))[0]

    def fit(self, x,y):
        self.x=x
        self.y=y
        w=[0 for i in range(np.shape(x)[1])]
        self.w=w
        b=0
        self.b=b
        cost_current=self.cost_func(w,b)
        say=0
        smm=0
        while   True:
            self.ws=[0 for _ in self.w]
            for ith in range(len(w)):
                self.ws[ith]= self.w[ith]- self.alpha*self.derivative(w=self.w,b=self.b,wth=ith,by="w")- \
                         ((self.alpha*self.lambda_)/(2*len(x)))*self.w[ith]
            self.b_new= self.b- self.alpha* self.derivative(w=self.w,b=self.b,by='b')
            self.w=self.ws
            self.b=self.b_new
            prt=[]
        # IF YOU WANT YOU CAN PLOT ALL POSSIBLE REGRESSIONS 
            #for x_i in x:
            #        prt.append(np.dot(w,x_i)+b)
            #plt.plot(x,prt,alpha=.1,c='red')
            
            cost_past=cost_current
            cost_current= self.cost_func(w,b)
        # IF YOU WANT YOU CAN PLOT CALCULATING GRADIENT DESCENT (ITS ERRORS)
            #plt.scatter(w[0],cost_current,c='purple')

            say+=1
            if smm<=10 and self.ws[0]<=0:
                self.alpha*=.1
                smm+=1
            if cost_past-cost_current<=self.alpha:
                say+=1
                if say==200:
                    break
            if cost_past-cost_current>self.alpha:
                say=0
            
        print('Ml model is ready')
    def predict(x):
        return (np.dot(x,np.reshape(w,(-1,1)))+b)





lr= LinearRegression(alpha=.001,lambda_=5)
lr.fit(x,y)
print("Prototype: ",np.sqrt(mean_squared_error(y,np.zeros((1,len(y)))[0])))
print('Model performance',np.sqrt(mean_squared_error(y,lr.predict(x))))









