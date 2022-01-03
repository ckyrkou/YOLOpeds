#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
# __/..(...|.(...|
# /__.-|_|--|_|
#
# Christos Kyrkou, PhD
# 2019

import numpy as np
import math

def lr_schedule_fix(epoch,lr):
    e=epoch+1

    e = np.mod(float(epochs), float(maxcount))

    if (e==0):
        lrate = lr/np.power(10,float(epoch)/100)


    return lrate


def lr_schedule_step(epoch, lr):
    lrate=lr
    if (epoch == 160):
        lrate = lr * 0.1
    if (epoch == 200):
        lrate = lr * 0.1
    return lrate

def lr_no_schedule(epoch,lr):
    return lr

def lr_schedule_cycle(epoch,lr,maxcount=3):

    e = np.mod(float(epoch),float(maxcount))

    lrate = 1e-4
    if (e==2):
        lrate = 1e-5


    return lrate

def cosine_decay(epochs_tot=500,initial_lrate=1e-1,warmup=False):
    def coside_decay_full(epoch,lr,epochs_tot=epochs_tot,initial_lrate=initial_lrate,warmup=warmup):

        lrate = 0.5 * (1 + math.cos(((epoch * math.pi) / (epochs_tot)))) * initial_lrate
        if(warmup and epoch <40):
            lrate = 1e-5
        if(lrate < 1e-5):
            lrate = 1e-5
        return lrate
    return coside_decay_full

def cosine_annealing(epochs_tot=500, eta_min=1e-6, eta_max=2e-4, T_max=10, fade=False,warmup=False):
    def coside_annealing_full(epoch, lr, epochs_tot=epochs_tot, eta_min=eta_min, eta_max=eta_max, T_max=T_max,fade=fade,warmup=warmup):
        lrate = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        # print((1.-float(epoch)/float(epochs_tot)))
        if(fade == True):
            lrate = lrate * (1. - 0.5*(float(epoch) / float(epochs_tot)))
        if(warmup and epoch<5):
            lrate=1e-6
        return lrate

    return coside_annealing_full
	
def SGDR(lr_max=1e-1, lr_min=1e-6, maxcount=10):
    def SGDR_fixed(epoch, lr,lr_max=lr_max, lr_min=lr_min, maxcount=maxcount):

        e = np.mod(float(epoch), float(maxcount))
        lrate = lr_min + 0.5 *(lr_max-lr_min)*(1 + math.cos(((e * math.pi) / (maxcount))))
        if(e == 0):
            maxcount = int(maxcount*2.)

        return lrate

    return SGDR_fixed
