#!/usr/bin/env python3
#
#  tcl_env.py
#  TCL environment for RL algorithms
#
# Author: Taha Nakabi

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import gym
# Trying out if this works for others. from gym import spaces had some issues
import gym.spaces as spaces

import math

# Default parameters for 
# default TCL environment.
# From Taha's code
DEFAULT_ITERATIONS = 24
DEFAULT_NUM_TCLS = 100
DEFAULT_NUM_LOADS = 150
# Load up default prices and 
# temperatures (from Taha's CSV)
default_data = np.load("default_price_and_temperatures.npy")
DEFAULT_PRICES = default_data[:,0]
DEFAULT_TEMPERATURS = default_data[:,1]
BASE_LOAD = np.array([2.0,2.0,2.0,2.0,3.4,4.0,6.0,5.5,6.0,5.5,4.0,3.3,4.1,3.3,4.1,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])
# https://austinenergy.com/ae/residential/rates/residential-electric-rates-and-line-items
PRICE_TIERS = np.array([2.8,5.8,7.8,9.3,10.81])

HIGH_PRICE_PENALTY = 2.0
FIXED_COST = 0
QUADRATIC_PRICE = .015

# Default Tmin and Tmax in TCLs
TCL_TMIN = 19
TCL_TMAX = 25
TCL_PENALTY=0.1
MAX_R = 1000
MAX_GENERATION = 120
SOCS_RENDER=[]
LOADS_RENDER =[]
BATTERY_RENDER = []
PRICE_RENDER = []
ENERGY_SOLD_RENDER = []
ENERGY_BOUGHT_RENDER = []
GRID_PRICES_RENDER = []
ENERGY_GENERATED_RENDER = []
TCL_CONTROL_RENDER=[]
TCL_CONSUMPTION_RENDER=[]
TOTAL_CONSUMPTION_RENDER=[]
ACTIONS = [[i,j,k,l] for i in range(4) for j in range(5) for k in range(2) for l in range(2)]



class TCL:
    """ 
    Simulates an invidual TCL
    """
    def __init__(self, ca, cm, q, P, Tmin=TCL_TMIN, Tmax=TCL_TMAX):
        self.ca = ca
        self.cm = cm
        self.q = q
        self.P = P
        self.Tmin = Tmin
        self.Tmax = Tmax

        # Added for clarity
        self.u = 0

    def set_T(self, T, Tm):
        self.T = T
        self.Tm = Tm

    def control(self, ui=0):
        # control TCL using u with respect to the backup controller
        if self.T < self.Tmin:
            self.u = 1
        elif self.Tmin<self.T<self.Tmax:
            self.u = ui
        else:
            self.u = 0

    def update_state(self, T0):
        # update the indoor and mass temperatures according to (22)
        for _ in range(5):
            self.T +=  self.ca * (T0 - self.T) + self.cm * (self.Tm - self.T) + self.P * self.u +self.q
            self.Tm += self.cm*(self.T - self.Tm)
            if self.T>=self.Tmax:
                 break

    """ 
    @property allows us to write "tcl.SoC", and it will
    run this function to get the value
    """
    @property
    def SoC(self):
        return (self.T-self.Tmin)/(self.Tmax-self.Tmin)

class Battery:
    # Simulates the battery system of the microGrid
    def __init__(self, capacity, useD, dissipation, lossC, rateC, maxDD, chargeE, tmax):
        self.capacity = capacity #full charge battery capacity
        self.useD = useD # useful discharge coefficient
        self.dissipation = dissipation # dissipation coefficient of the battery
        self.lossC = lossC #charge loss
        self.rateC = rateC #charging rate
        self.maxDD = maxDD #maximum power that the battery can deliver per timestep
        self.tmax= tmax #maxmum charging time
        self.chargeE = chargeE #Energy given to the battery to charge
        self.RC = 0 #remaining capacity
        self.ct = 0 #Charging step

    def charge(self, E):
        empty = self.capacity-self.RC
        if empty <= 0:
            return E
        else:
            self.RC += self.rateC*E
            leftover = self.RC - self.capacity
            self.RC = min(self.capacity,self.RC)
            return max(leftover,0)


    def supply(self, E):
        remaining = self.RC
        self.RC-= E*self.useD
        self.RC = max(self.RC,0)
        return min(E, remaining)

    def dissipate(self):
        self.RC = self.RC * math.exp(- self.dissipation)

    @property
    def SoC(self):
        return self.RC/self.capacity

class Grid:
    def __init__(self):
        down_reg_df=pd.read_csv("down_regulation.csv")
        up_reg_df = pd.read_csv("up_regulation.csv")
        down_reg = np.array(down_reg_df.iloc[:,-1])/10
        up_reg = np.array(up_reg_df.iloc[:, -1])/10
        self.buy_prices = down_reg
        self.sell_prices = up_reg
        self.time = 0

    def sell(self, E):
        return self.sell_prices[self.time]*E

    def buy(self,E):
        return -self.buy_prices[self.time]*E - QUADRATIC_PRICE*E**2 - FIXED_COST
    #
    # def get_price(self,time):
    #     return self.prices[time]

    def set_time(self,time):
        self.time = time

    def total_cost(self,prices, energy):
        return sum(prices*energy/100+ QUADRATIC_PRICE*energy**2 - FIXED_COST)


class Generation:
    def __init__(self, max_capacity):
        power_df = pd.read_csv("wind_generation.csv")
        self.power = np.array(power_df.iloc[:,-1])
        self.max_capacity = np.max(self.power[:30])

    def current_generation(self,time):
        # We consider that we have 2 sources of power a constant source and a variable source
        return  self.power[time]


class Load:
    def __init__(self, price_sens, base_load, max_v_load):
        self.price_sens = price_sens
        self.base_load = base_load
        self.max_v_load = max_v_load
        self.response = 0

    def react(self, price_tier):
        self.response = self.price_sens*(price_tier-2)
        if self.response > 0 and self.price_sens > 0.1:
            self.price_sens-= 0.1

    def load(self, time_day):
        # print(self.response)
        return max(self.base_load[time_day] - self.max_v_load*self.response,0)



class MicroGridEnv(gym.Env):
    def __init__(self, **kwargs):
        """
        Arguments:
            iterations: Number of iterations to run
            num_tcls: Number of TCLs to create in cluster
            prices: Numpy 1D array of prices at different times
            temperatures : Numpy 1D array of temperatures at different times
        """

        # Get number of iterations and TCLs from the 
        # parameters (we have to define it through kwargs because 
        # of how Gym works...)
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.num_tcls = kwargs.get("num_tcls", DEFAULT_NUM_TCLS)
        self.num_loads = kwargs.get("num_loads", DEFAULT_NUM_LOADS)
        self.prices = kwargs.get("prices", DEFAULT_PRICES)
        self.temperatures = kwargs.get("temperatures", DEFAULT_TEMPERATURS)
        self.base_load = kwargs.get("base_load", BASE_LOAD)
        self.price_tiers = kwargs.get("price_tiers", PRICE_TIERS)

        # The current day: pick randomly
        self.day = random.randint(0,10)
        # self.day = 8
        # self.day = 55
        # The current timestep
        self.time_step = 0

        # The cluster of TCLs to be controlled.
        # These will be created in reset()
        self.tcls_parameters = []
        self.tcls = []
        # The cluster of loads.
        # These will be created in reset()
        self.loads_parameters = []
        self.loads = []

        self.generation = Generation(MAX_GENERATION)
        self.grid = Grid()

        for i in range(self.num_tcls):
            self.tcls_parameters.append(self._create_tcl_parameters())

        for i in range(self.num_loads):
            self.loads_parameters.append(self._create_load_parameters())

        self.action_space = spaces.Discrete(80)
        
        # Observations: A vector of TCLs SoCs + loads +battery soc+ power generation + price + temperature + time of day
        self.observation_space = spaces.Box(low=-100, high=100, dtype=np.float32, 
                    shape=(1  + 7,))


    def _create_tcl_parameters(self):
        """
                Initialize one TCL randomly with given T_0,
                and return it. Copy/paste from Taha's code
                """
        # Hardcoded initialization values to create
        # bunch of different TCLs
        ca = random.normalvariate(0.004, 0.0008)
        cm = random.normalvariate(0.3, 0.004)
        q = random.normalvariate(0, 0.01)
        P = random.normalvariate(1.5, 0.01)
        return [ca,cm,q,P]

    def _create_tcl(self,ca ,cm ,q ,P, initial_temperature):
        tcl= TCL(ca,cm,q,P)
        tcl.set_T(initial_temperature,initial_temperature)
        return tcl
    def _create_load_parameters(self):

        """
        Initialize one load randomly,
        and return it.
        """
        # Hardcoded initialization values to create
        # bunch of different loads

        price_sensitivity= random.normalvariate(0.5, 0.3)
        max_v_load = random.normalvariate(3.0, 1.0)
        return [price_sensitivity,max_v_load]

    def _create_load(self,price_sensitivity,max_v_load):
        load = Load(price_sensitivity,base_load=self.base_load, max_v_load=max_v_load)
        return load

    def _create_battery(self):
        """
        Initialize one battery
        """
        battery = Battery(capacity = 400.0, useD=0.9, dissipation=0.001, lossC=0.15, rateC=0.9, maxDD=10, chargeE=10, tmax=5)
        return battery

    def _build_state(self):
        """ 
        Return current state representation as one vector.
        Returns:
            state: 1D state vector, containing state-of-charges of all TCLs, Loads, current battery soc, current power generation,
                   current temperature, current price and current time (hour) of day
        """
        # SoCs of all TCLs binned + current temperature + current price + time of day (hour)
        socs = np.array([tcl.SoC for tcl in self.tcls])
        # Scaling between -1 and 1
        socs = (socs+np.ones(shape=socs.shape)*4)/(1+4)
        socs=np.average(socs)

        # loads = np.array([l.load(self.time_step) for l in self.loads])
        # loads = sum([l.load(self.time_step) for l in self.loads])
        # # Scaling loads
        # loads = (loads-(min(BASE_LOAD)+2)*DEFAULT_NUM_LOADS)/((max(BASE_LOAD)+4-min(BASE_LOAD)-2)*DEFAULT_NUM_LOADS)

        loads = BASE_LOAD[(self.time_step) % 24]
        loads = (loads - min(BASE_LOAD)) / (max(BASE_LOAD) - min(BASE_LOAD))


        current_generation = self.generation.current_generation(self.day+self.time_step)
        current_generation /= self.generation.max_capacity

        temperature = self.temperatures[self.day+self.time_step]
        temperature = (temperature-min(self.temperatures))/(max(self.temperatures)-min(self.temperatures))

        price = self.grid.buy_prices[self.day+self.time_step]
        price = (price - min(self.grid.buy_prices)) / (max(self.grid.buy_prices) - min(self.grid.buy_prices))

        high_price = self.high_price/(4 * self.iterations)

        time_step = (self.time_step)/24

        state = np.array([socs, loads, high_price, self.battery.SoC, current_generation,
                         temperature,
                         price,
                         time_step])
        return state

    def _build_info(self):
        """
        Return dictionary of misc. infos to be given per state.
        Here this means providing forecasts of future
        prices and temperatures (next 24h)
        """
        temp_forecast = np.array(self.temperatures[self.time_step+1:self.time_step+25])
        price_forecast = np.array(self.prices[self.time_step+1:self.time_step+25])
        return {"temperature_forecast": temp_forecast, 
                "price_forecast": price_forecast,
                "forecast_times": np.arange(0,self.iterations)}

    
    def _compute_tcl_power(self):
        """
        Return the total power consumption of all TCLs
        """
        return sum([tcl.u*tcl.P for tcl in self.tcls])

    def step(self, action):
        """ 
        Arguments:
            action: A list.
        
        Returns:
            state: Current state
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (maximum number of iterations)
            info: None (not used here)
        """
        if type(action) is not list:
            action = ACTIONS[action]

        self.grid.set_time(self.day+self.time_step)
        reward = 0
        # Update state of TCLs according to action


        tcl_action = action[0]
        price_action = action[1]
        energy_deficiency_action = action[2]
        energy_excess_action = action[3]
        # Get the energy generated by the DER
        available_energy = self.generation.current_generation(self.day+self.time_step)
        # Energy rate
        # self.eRate = available_energy/self.generation.max_capacity

        # print("Generated power: ", available_energy)
        # We implement the pricing action and we calculate the total load in response to the price
        for load in self.loads:
            load.react(price_action)
        total_loads = sum([l.load(self.time_step) for l in self.loads])
        # print("Total loads",total_loads)
        # We fulfilled the load with the available energy.
        available_energy -= total_loads
        # We calculate the return based on the sale price.
        self.sale_price = self.price_tiers[price_action]
        # We increment the reward by the amount of return
        # Division by 100 to transform from cents to euros
        reward += total_loads*self.sale_price/100
        # Penalty of charging too high prices
        self.high_price += price_action
        # Distributing the energy according to priority
        sortedTCLs = sorted(self.tcls, key=lambda x: x.SoC)
        # print(tcl_action)
        control = tcl_action*50.0
        self.control = control
        for tcl in sortedTCLs:
            if control>0:
                tcl.control(1)
                control-= tcl.P * tcl.u
            else:
                tcl.control(0)
            tcl.update_state(self.temperatures[self.day+self.time_step])
            # if tcl.SoC >1 :
            #     reward -= abs((tcl.SoC-1) * reward*TCL_PENALTY)
            # if  tcl.SoC<0:
            #     reward += tcl.SoC * abs(reward*TCL_PENALTY)

        available_energy -= self._compute_tcl_power()
        # control_error = self.sale_price*(self.control-self._compute_tcl_power())**2
        reward += self._compute_tcl_power()*self.sale_price/100
        if available_energy>0:
            if energy_excess_action:
                available_energy = self.battery.charge(available_energy)
                reward += self.grid.sell(available_energy)/100
            else:
                reward += self.grid.sell(available_energy)/100
            self.energy_sold =  available_energy
            self.energy_bought = 0

        else:
            if energy_deficiency_action:
                available_energy += self.battery.supply(-available_energy)

            self.energy_bought = -available_energy
            reward += self.grid.buy(self.energy_bought)/100
            self.energy_sold = 0

        # Proceed to next timestep.
        self.time_step += 1
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state()

        if self.high_price > 2 * self.iterations :
            # Penalize high prices
            reward -= abs(reward * HIGH_PRICE_PENALTY * (self.high_price - 2 * self.iterations ))
        terminal = self.time_step == self.iterations - 1
        if terminal:
            # reward if battery is charged
            reward += abs(reward*self.battery.SoC/2)
        info = self._build_info()

        return state, reward/MAX_R ,terminal, info

    def reset(self,day=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if day==None:
            self.day = random.randint(0,10)
        else:
            self.day = day
        print("Day:",self.day)
        self.time_step = 0
        self.battery = self._create_battery()
        self.energy_sold = 0
        self.energy_bought = 0
        self.energy_generated = 0
        self.control=0
        self.sale_price = PRICE_TIERS[2]
        self.high_price = 0
        self.tcls.clear()
        # initial_tcl_temperature = random.normalvariate(12, 5)
        initial_tcl_temperature = 12

        for i in range(self.num_tcls):
            parameters = self.tcls_parameters[i]
            self.tcls.append(self._create_tcl(parameters[0],parameters[1],parameters[2],parameters[3],initial_tcl_temperature))

        self.loads.clear()
        for i in range(self.num_loads):
            parameters = self.loads_parameters[i]
            self.loads.append(self._create_load(parameters[0],parameters[1]))

        self.battery = self._create_battery()
        return self._build_state()

    def render(self,name=''):
        SOCS_RENDER.append([tcl.SoC for tcl in self.tcls])
        LOADS_RENDER.append([l.load(self.time_step) for l in self.loads])
        PRICE_RENDER.append(self.sale_price)
        BATTERY_RENDER.append(self.battery.SoC)
        ENERGY_GENERATED_RENDER.append(self.generation.current_generation(self.day+self.time_step))
        ENERGY_SOLD_RENDER.append(self.energy_sold)
        ENERGY_BOUGHT_RENDER.append(self.energy_bought)
        GRID_PRICES_RENDER.append(self.grid.buy_prices[self.day+self.time_step])
        TCL_CONTROL_RENDER.append(self.control)
        TCL_CONSUMPTION_RENDER.append(self._compute_tcl_power())
        TOTAL_CONSUMPTION_RENDER.append(self._compute_tcl_power()+np.sum([l.load(self.time_step) for l in self.loads]))
        if self.time_step==self.iterations-1:
            # fig=pyplot.figure()
            # ax1 = fig.add_subplot(3,3,1)
            # ax1.boxplot(np.array(SOCS_RENDER).T)
            # ax1.set_title("TCLs SOCs")
            # ax1.set_xlabel("Time (h)")
            # ax1.set_ylabel("SOC")
            #
            # ax2 = fig.add_subplot(3, 3, 2)
            # ax2.boxplot(np.array(LOADS_RENDER).T)
            # ax2.set_title("LOADS")
            # ax2.set_xlabel("Time (h)")
            # ax2.set_ylabel("HOURLY LOADS")
            #
            # ax3 = fig.add_subplot(3, 3, 3)
            # ax3.plot(PRICE_RENDER)
            # ax3.set_title("SALE PRICES")
            # ax3.set_xlabel("Time (h)")
            # ax3.set_ylabel("HOURLY PRICES")
            #
            # ax4 = fig.add_subplot(3, 3, 4)
            # ax4.plot(np.array(BATTERY_RENDER))
            # ax4.set_title("BATTERY SOC")
            # ax4.set_xlabel("Time (h)")
            # ax4.set_ylabel("BATTERY SOC")
            #
            # ax4 = fig.add_subplot(3, 3, 5)
            # ax4.plot(np.array(ENERGY_GENERATED_RENDER))
            # ax4.set_title("ENERGY_GENERATED")
            # ax4.set_xlabel("Time (h)")
            # ax4.set_ylabel("ENERGY_GENERATED")
            #
            # ax4 = fig.add_subplot(3, 3, 6)
            # ax4.plot(np.array(ENERGY_SOLD_RENDER))
            # ax4.set_title("ENERGY_SOLD")
            # ax4.set_xlabel("Time (h)")
            # ax4.set_ylabel("ENERGY_SOLD")
            #
            # ax4 = fig.add_subplot(3, 3, 7)
            # ax4.plot(np.array(ENERGY_BOUGHT_RENDER))
            # ax4.set_title("ENERGY_BOUGHT")
            # ax4.set_xlabel("Time (h)")
            # ax4.set_ylabel("ENERGY_BOUGHT")
            #
            # ax4 = fig.add_subplot(3, 3, 8)
            # ax4.plot(np.array(GRID_PRICES_RENDER))
            # ax4.set_title("GRID_PRICES")
            # ax4.set_xlabel("Time (h)")
            # ax4.set_ylabel("GRID_PRICES_RENDER")
            #
            # ax4 = fig.add_subplot(3, 3, 9)
            # ax4.bar(x=np.array(np.arange(self.iterations)),height=TCL_CONTROL_RENDER,width=0.2)
            # ax4.bar(x=np.array(np.arange(self.iterations))+0.2,height=TCL_CONSUMPTION_RENDER,width=0.2)
            # ax4.set_title("TCL_CONTROL VS TCL_CONSUMPTION")
            # ax4.set_xlabel("Time (h)")
            # ax4.set_ylabel("kW")
            # pyplot.show()
            np.save(name + 'Cost' + str(self.day) + '.npy', self.grid.total_cost(np.array(GRID_PRICES_RENDER),np.array(ENERGY_BOUGHT_RENDER)))
            np.save(name + 'Energy_bought_sold' + str(self.day) + '.npy', np.array(ENERGY_BOUGHT_RENDER)-np.array(ENERGY_SOLD_RENDER))
            np.save(name+'TOTAL_Consumption'+str(self.day)+'.npy' , TOTAL_CONSUMPTION_RENDER)
            SOCS_RENDER.clear()
            LOADS_RENDER.clear()
            PRICE_RENDER.clear()
            BATTERY_RENDER.clear()
            GRID_PRICES_RENDER.clear()
            ENERGY_BOUGHT_RENDER.clear()
            ENERGY_SOLD_RENDER.clear()
            ENERGY_GENERATED_RENDER.clear()
            TCL_CONTROL_RENDER.clear()
            TCL_CONSUMPTION_RENDER.clear()
            TOTAL_CONSUMPTION_RENDER.clear()


    def close(self):
        """ 
        Nothing to be done here, but has to be defined 
        """
        return

    def seed(self, seed):
        """
        Set the random seed for consistent experiments
        """
        random.seed(seed)
        np.random.seed(seed)
        
if __name__ == '__main__':
    # Testing the environment
    from matplotlib import pyplot
    # Initialize the environment
    env = MicroGridEnv()
    env.seed(1)
    # Save the rewards in a list
    rewards = []
    # reset the environment to the initial state
    state = env.reset()
    # Call render to prepare the visualization
    env.render()
    # Interact with the environment (here we choose random actions) until the terminal state is reached
    while True:
        # Pick an action from the action space (here we pick an index between 0 and 80)
        action = env.action_space.sample()
        # Using the index we get the actual action that we will send to the environment
        print(ACTIONS[action])
        # Perform a step in the environment given the chosen action
        state, reward, terminal, _ = env.step(action)
        env.render()
        print(reward)
        rewards.append(reward)
        if terminal:
            break
    print("Total Reward:",sum(rewards))

    # Plot the TCL SoCs 
    states = np.array(rewards)
    pyplot.plot(rewards)
    pyplot.title("rewards")
    pyplot.xlabel("Time")
    pyplot.ylabel("rewards")
    pyplot.show()

