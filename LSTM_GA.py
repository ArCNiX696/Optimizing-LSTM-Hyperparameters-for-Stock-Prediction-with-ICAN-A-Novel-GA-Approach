import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchmetrics import MeanAbsolutePercentageError , R2Score
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import gc
import os
import time

i = [0]
t_counter = 1

def create_folders(folder,folder_2, sub_folder, sub_folder_2,sub_folder_3):
    n = i[-1]  # Último elemento de la lista `i`
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    if not os.path.exists(folder_2):
        os.makedirs(folder_2)

    while os.path.exists(sub_folder) or os.path.exists(sub_folder_2)or os.path.exists(sub_folder_3):
        n += 1
        sub_folder = f'./Statistics results/Training and validation losses/LSTM GA Run {n}'
        sub_folder_2 = f'./Statistics results/GA statistics/LSTM GA Run {n}'
        sub_folder_3 = f'./Best model weights/LSTM GA Run {n}'
    
    os.makedirs(sub_folder)
    os.makedirs(sub_folder_2)
    os.makedirs(sub_folder_3)
    i.append(n)  

    return sub_folder, sub_folder_2,sub_folder_3


directory = './Statistics results'
directory_2 = './Best model weights/'
sub_directory, sub_directory_2 ,sub_directory_3= create_folders(directory,directory_2,
                                                f'./Statistics results/Training and validation losses/LSTM GA Run {i[-1]}',
                                                f'./Statistics results/GA statistics/LSTM GA Run {i[-1]}',
                                                f'./Best model weights/LSTM GA Run {i[-1]}/')


device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SimpleGA():
    def __init__(self, population_size=int, chromosome_len=int,generations=int,tournament_size=int,crossover_rate=float,inherited_rate=float,mutation_rate=float):
        self.population_size = population_size
        self.chromosome_len = chromosome_len
        self.generations = generations
        # self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.inherited_rate= inherited_rate
        self.mutation_rate = mutation_rate
        self.best_chromosome = None
        self.best_generation = None
        self.best_fitness = []
        self.best_fitness_ican = []
        # self.actual_fitness = []
        # self.standar_devs = []

    ############## GA Utils ###############

    def chromosome_partition(self,chromosome,verbose=False):
   
        chromosome_len=len(chromosome)
        
        part_1 = chromosome[:int(chromosome_len - 2)]
    
        hidden_size = part_1[:3]
    
        staked_layers = part_1[3:6]

        learning_rate = part_1[6:8]

        batch_size = part_1[8:11]

        epochs = part_1[11:15]

        #part_2
        split = chromosome[int(chromosome_len - 2):]

        if verbose == True:
            print(f'\n{"*" * 120}\n')
            print(f'chromosome : {chromosome} , len : {len(chromosome)}\n')
            print(f'part_1 : {part_1} , len : {len(part_1)}\n')
            print(f'hidden_size : {hidden_size} , len : {len(hidden_size)}\n')
            print(f'staked_layers : {staked_layers} , len : {len(staked_layers)}\n')
            print(f'batch_size : {batch_size} , len : {len(batch_size)}\n')
            print(f'epochs : {epochs} , len : {len(epochs)}\n')
            print(f'part_2 split train data: {split} , len : {len(split)}\n') 
            print(f'\n{"*" * 120}\n')
        
        return hidden_size,staked_layers,learning_rate,batch_size,epochs,split

    def decode(self,bits,values=list):
        #Convert the bits of the chrmosome in a binary string
        bit_str = ''.join(str(bit) for bit in bits)

        decimal_val = int(bit_str,2)#then that binary string to decimal

        decode_chro = values[decimal_val]

        return decode_chro

    #Separeate each chromosome into 2 binary chro
    def decode_partitions(self,hidden_size,staked_layers,learning_rate,batch_size,epochs,split,verbose=False):
        
        hidden_size_values = [1,2,3,4,5,6,7,8]
        
        staked_layers_values = [1,2,3,4,5,6,7,8]

        learning_rate_values = [1e-2, 1e-3, 1e-4, 1e-5]

        batch_size_values = [8, 16, 32, 64, 128, 256, 512, 1024]

        epochs_values = [500, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

        # epochs_values = [5, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]

        split_values = [0.6, 0.7, 0.8 , 0.9]
        
        hidden_size = self.decode(hidden_size,hidden_size_values)

        staked_layers = self.decode(staked_layers,staked_layers_values)

        learning_rate = self.decode(learning_rate,learning_rate_values)

        batch_size = self.decode(batch_size,batch_size_values)

        epochs = self.decode(epochs,epochs_values)

        #part_2
        split = self.decode(split,split_values)

        if verbose == True:
            print(f'hidden_size decoded : {hidden_size}\n')
            print(f'staked_layers decoded : {staked_layers}\n')
            print(f'batch_size decoded : {batch_size}\n')
            print(f'epochs decoded : {epochs}\n')
            print(f'split train data : {split}\n')
            print(f'\n{"*" * 120}\n')

        return hidden_size,staked_layers,learning_rate,batch_size,epochs,split
    
    def evaluate_fitness(self,individual):
        hidden_size, staked_layers, learning_rate, batch_size, epochs, split = self.chromosome_partition(individual, verbose=False)
        hidden_size, staked_layers, learning_rate, batch_size, epochs, split = self.decode_partitions(hidden_size, staked_layers, learning_rate, batch_size, epochs, split, verbose=True)
        
        model = StockPreLSTM(hidden_size, staked_layers, learning_rate, batch_size, epochs, split)
        best_loss = model.run_model()

        # Liberar memoria
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return best_loss
    
    def plot_evolution(self,data,x_label=str,y_label=str,title=str,file_name=str):
        plt.figure(figsize=(8, 8))  # Definir el tamaño de la figura

        # Graficar los datos
        plt.plot(data, color='blue',marker = 'o', linewidth=2)

        # Etiquetas y título
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Mostrar la gráfica
        plt.grid(True)  # Agregar cuadrícula
        plot_path = os.path.join(sub_directory_2,f'{file_name}.png')
        plt.savefig(plot_path)
        plt.close() 
        gc.collect()
        # plt.show()

    def seconds_to_hours(self,seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

    ############## GA Simple ###############

    def initialize_population(self,verbose=False):
        population = np.random.randint(2,size=(self.population_size,self.chromosome_len))

        if verbose==True:
            np.set_printoptions(threshold=np.inf)#Set print options to display all elements
            print(f'population: {population}')

        return population
    
    #Selection
    def tournament(self,population,tounament_rate,fitness_val,verbose=False):
        n=len(population)
        print(f'n {n} = {n}')
        winners=[]

        for i in range(n):
            tournament_idx=np.random.choice(np.arange(n),size=tounament_rate,replace=False)
            # print(f'tournamen idx {i} = {tournament_idx}')

            tournament_pop=[population[idx] for idx in tournament_idx]
            # print(f'tournament_pop  = {tournament_pop}')
            
            winner= min(tournament_pop,key =fitness_val )
            winners.append(winner)

        if verbose==True:
            print('-'*100,'\n')
            print(f'Winners of the touenament : \n{winners}')
            print('-'*100,'\n')
            # print(f'tournamen pop {i} = {tournament_pop}')
            # print(f'The winer in  tournament {i} {tournament_idx}is = {winner}')

        return winners
    
    #Crossover
    def uniform_crossover(self,parent1,parent2,cross_rate,inherited_rate,verbose=False):
        parent1_bits=[]
        parent2_bits=[]
        crossover=False

        if not isinstance(cross_rate, float):
            return print(f'\ncross_rate = {cross_rate},cross_rate debe ser un valor float\n')
            
    
        if np.random.rand()<cross_rate:
            crossover=True
            
            child1=np.empty_like(parent1)
            child2=np.empty_like(parent2)

            for i in range(len(parent1)):
                prob=np.random.rand()
                
                if prob < inherited_rate:
                    child1[i]=parent1[i]
                    child2[i]=parent1[i]

                    parent1_bits.append(i)

                else:
                    child1[i]=parent2[i]
                    child2[i]=parent1[i]

                    parent2_bits.append(i)

        else:
            child1=parent1.copy()
            child2=parent2.copy()

        if verbose==True and crossover==True:
            print('-'*100,'\n')
            print(f'\nParent1 = {parent1}\nParent2 = {parent2}\n')
            print(f'Bits inherited by\nparent 1 are in indexes = {parent1_bits}\nparent 2 are in indexes = {parent2_bits}\n')
            print(f'child1 = {child1}\nchild2 = {child2}\n')
            print('-'*100,'\n')

        elif verbose == False:
            pass

        else:
            print('-'*100,'\n')
            print(f'\nParent1 = {parent1}\nParent2 = {parent2}\n')
            print(f'\nNo crossover in this generation due a low crossover rate, so parents become children for the next generation\n')
            print(f'child1 = {child1}\nchild2 = {child2}')
            print('-'*100,'\n')
            
        return child1,child2


    def inter_onepoint_cross(self,parent1,parent2,cross_rate,verbose=False):
        n=len(parent1)
        m=len(parent2)
        

        if np.random.rand() < cross_rate:
            cross_point1=np.random.randint(1,n)
            cross_point2=np.random.randint(1,m)

            child1=np.concatenate((parent1[cross_point1:],parent1[:cross_point1]))
            child2=np.concatenate((parent2[cross_point2:],parent2[:cross_point2]))
        
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        if verbose == True:
            if verbose==True:
                print('-'*100,'\n')
                print(f'\nParent1 = {parent1}\nParent2 = {parent2}\n')
                print(f'crosover_point 1 for child 1 = {cross_point1}\n')
                print(f'crosover_point 2 for child 2 = {cross_point2}\n')
                print(f'child1 = {child1}\nchild2 = {child2}')
                print('-'*100,'\n')

        return child1,child2

    def bitflip_mutation(self,chromosome,mut_rate,verbose=False):
        mutation=False
        mutated_bits=[]
        mutated_chro= chromosome.copy()

        for i in range(len(chromosome)):
            if np.random.rand()<mut_rate:
                mutation=True
                
                bit=chromosome[i]
                mutated_bit = 1 - bit
                mutated_chro[i]=mutated_bit

                #For debugging
                # print('bit:',bit)
                # print('mutated bit :',mutated_bit)

                mutated_bits.append(i)

            else:
                mutated_chro[i] = mutated_chro[i]
            
        if verbose==True and mutation == True:
            print('-'*100,'\n')
            print(f'Original Chromosome = {chromosome}\n\n')
            print(f'The mutated genes of this chromosome are in indexes = {mutated_bits}\n\n')
            print(f'Chromosome after mutation = {mutated_chro}\n\n')        
            print('-'*100,'\n')

        elif verbose==False:
            pass

        else:
            print(f'Original Chromosome = {chromosome}\n\n')
            print(f'Due the low mutation rate, there is not mutation for this Chromosome in this generation\n\n')
            print(f'Chromosome ramains the same = {mutated_chro}\n\n')    

        return mutated_chro

    def simpleGA(self,verbose=False,plot=False):
        start_time = time.time()
        population = self.initialize_population(verbose=False)
        best_fitness=float('inf')
        trace_fitnness = []
        indi_counter = 0
        
        for generation in range(self.generations):
            print(f'\n{"-"* 100}\n')
            print(f'\ngeneration  : {generation}\n')

            for individual in population:
                indi_counter +=1
                print(f'\nindividual  : {individual}\n')
                selected_pop = self.tournament(population, self.tournament_size, fitness_val=self.evaluate_fitness)
                new_population = []
                
                for i in range(0, len(selected_pop)-1, 2):
                    parent1, parent2 = selected_pop[i], selected_pop[i+1]
                    child1, child2 = self.uniform_crossover(parent1, parent2, self.crossover_rate, self.inherited_rate)
                    child1 = self.bitflip_mutation(child1,self.mutation_rate)
                    child2= self.bitflip_mutation(child2,self.mutation_rate)
                    
                    new_population.extend([child1, child2])

            population = new_population
        
            chromosome_fitness_pairs = [(chromosome, self.evaluate_fitness(chromosome)) for chromosome in population]
            current_best_chromosome, current_best_fitness = min(chromosome_fitness_pairs, key=lambda x: x[1])

            # current_best_chromosome = min(population,key=self.evaluate_fitness)
            # current_best_fitness = abs(self.evaluate_fitness(current_best_chromosome))
            self.best_fitness.append(current_best_fitness)
            hidden_size,staked_layers,learning_rate,batch_size,epochs,split = self.chromosome_partition(current_best_chromosome)
            best_solution = self.decode_partitions(hidden_size,staked_layers,learning_rate,batch_size,epochs,split)

            #Track the generations
            self.plot_evolution(self.best_fitness,'Number of Generations','Fitness score','Chromosome evolution in all generations',f'Evolution till generation {generation}, best fitness {best_fitness}')
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                trace_fitnness.append(best_fitness)
                self.best_chromosome = current_best_chromosome
                self.best_generation = generation

            
            if verbose == True:
                print(f'Generation : {generation}\n')
                print(f'Best solution :\nBinary chromosome = {current_best_chromosome}')
                print(f'\nDecoded best solution :\nhidden_size = {best_solution[0]}\nstaked_layers = {best_solution[1]}\nlearning_rate = {best_solution[2]}\nbatch_size = {best_solution[3]}\nepochs = {best_solution[4]},\nsplit = {best_solution[5]}')
                print(f'\nFitness score = {current_best_fitness}')
                print('-'*100,'\n')
            

        if verbose == True:
                hidden_size,staked_layers,learning_rate,batch_size,epochs,split = self.chromosome_partition(self.best_chromosome)
                best_global_solution = self.decode_partitions(hidden_size,staked_layers,learning_rate,batch_size,epochs,split)
                
                print(f'Best solution was founded in generation : {self.best_generation}\n')
                print(f'Best solution :\nBinary chromosome = {self.best_chromosome}')
                print(f'\nFitness score = {best_fitness}')
                print(f'\nDecoded best solution :\nhidden_size = {best_global_solution[0]}\nstaked_layers = {best_global_solution[1]}\nlearning_rate = {best_global_solution[2]}\nbatch_size = {best_global_solution[3]}\nepochs = {best_global_solution[4]},\nsplit = {best_global_solution[5]}')
                print(f'\nAll best fitness scores = {trace_fitnness}')
                print('-'*100,'\n')

        if plot == True:
           self.plot_evolution(self.best_fitness,'Number of Generations','Fitness score','Chromosome evolution in all generations',f'Evolution in {self.generations} generations, best fitness {best_fitness}')

        
        fitness_reg = f'Best fitness register.txt'
        with open(os.path.join(sub_directory_2, fitness_reg), 'a') as log_file:
            log_file.write(f'Best fitness list : {self.best_fitness}\n')
        
        elapsed_time = time.time() - start_time
        elapsed_time_str = self.seconds_to_hours(elapsed_time)
        with open(os.path.join(sub_directory_2, fitness_reg), 'a') as log_file:
            log_file.write(f'Total time: {elapsed_time_str}\n')

class PrepareDataset:
    def __init__(self):
        raw_folder='./input datasets/'
        # raw_data='TESLA stock price dataset - windows size of  t-14 days.csv'
        raw_data='TESLA stock price dataset - windows size of  t-7 days,Normalized.csv'
        self.path=os.path.join(raw_folder,raw_data)
        
    def import_dataset(self,verbose=False):
        self.df = pd.read_csv(self.path )

        if verbose == True:
            print(f'\n{self.df.head(10)}\n') 
        
        return self.df
    
    def split_data(self,split_train=float,verbose=False):
        #Import dataset
        self.df = self.import_dataset(verbose=False)

        if self.df is None:
            print('No dataset availiable for split')

        else:
            self.df = self.df.to_numpy()

            x = self.df[:,1:]
            x= dc(np.flip(x, axis= 1))#For LSTM should flip the dataset to change the order in time 
            x_col_size = x.shape[1]

            y = self.df[:,0]

        if len(x) != len (y):
            print('x  has not the same number of rows as y , then cannot split data')

        else :
            split_idx = int(len(x) * split_train)

            #Train partition
            x_train = x[:split_idx].reshape((-1 , x_col_size , 1))
            x_test = x[split_idx:].reshape((-1 , x_col_size , 1))
            #Train partition to tensor
            x_train = torch.tensor(x_train).float()
            x_test = torch.tensor(x_test).float()

            #Test partition
            y_train = y[:split_idx].reshape((-1 , 1))
            y_test = y[split_idx:].reshape((-1 , 1))
            #Train partition to tensor
            y_train = torch.tensor(y_train).float()
            y_test = torch.tensor(y_test).float()

        
        if verbose == True:
            print(f'\n{"-"* 100}\n')
            print(f'\nx dataset with type and shape : {x.dtype} , {x.shape} len {len(x)}:\n\n{x}\n') 
            print(f'\n{"-"* 100}\n')
            print(f'\ny dataset type and shape : {y.dtype} , {y.shape}  len {len(y)}:\n\n{y}\n') 
            print(f'\n{"-"* 100}\n')
            print(f'\nx_train dataset type and shape : {x_train.dtype} , {x_train.shape}  len {len(x_train)}:\n\n{x_train}\n') 
            print(f'\nx_test dataset type and shape : {x_test.dtype} , {x_test.shape}  len {len(x_test)}:\n\n{x_test}\n')
            print(f'\n{"-"* 100}\n')
            print(f'\ny_train dataset type and shape : {y_train.dtype} , {y_train.shape}  len {len(y_train)}:\n\n{y_train}\n') 
            print(f'\ny_test dataset type and shape : {y_test.dtype} , {y_test.shape}  len {len(y_test)}:\n\n{y_test}\n')

        return x_train ,x_test ,y_train , y_test


class CustomDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i],self.y[i]
    

class LstmStructure(nn.Module,SimpleGA):
    def __init__(self, input_size , hidden_size ,staked_layers): #input_size = num of features , hidden_size = num of memory_cells ,staked_layers = num of layers in LSTM
        super().__init__()
        SimpleGA.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stacked_layers = staked_layers
        
        self.lstm = nn.LSTM(input_size,hidden_size,staked_layers,batch_first=True)
        self.fully_conected = nn.Linear(hidden_size , 1)

    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.stacked_layers, batch_size , self.hidden_size).to(device) #hidden state
        c0 = torch.zeros(self.stacked_layers, batch_size , self.hidden_size).to(device) #cell state

        out,_=self.lstm(x , (h0,c0))
        out = self.fully_conected(out[:,-1,:])
        return out 

class StockPreLSTM(nn.Module):
    def __init__(self,hidden_size,staked_layers,learning_rate,batch_size,epochs,split):
        super().__init__()
        self.hidden_size = hidden_size
        self.staked_layers = staked_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.split = split
        self.model=LstmStructure(1,self.hidden_size,self.staked_layers).to(device) #input_size = num of features , hidden_size = num of memory_cells ,staked_layers = num of layers in LSTM
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.MAPE = MeanAbsolutePercentageError().to(device)
        self.R2 = R2Score().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.best_loss = float('inf')
        self.best_model_path = sub_directory_3
        self.training_losses = []
        self.validation_losses = []
        # self.counter = t_counter

    
    def prepare_data(self,verbose=False):
        data = PrepareDataset()

        x_train ,x_test ,y_train , y_test = data.split_data(self.split) #Split data

        train_dataset = CustomDataset(x_train , y_train)
        self.train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)

        val_dataset = CustomDataset(x_test, y_test)
        self.val_loader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False)

        if verbose == True:
            for _, b in enumerate(self.train_loader):
                x_b , y_b = b[0].to(device) , b[1].to(device)
                print(f'x_b shape: {x_b.shape} ,y_b shape: {y_b.shape}')
                break

        return self.train_loader , self.val_loader
    
    def training_model(self):
        global t_counter

        for self.epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs,labels in tqdm(self.train_loader,desc=f'Epoch{self.epoch+1}/{self.epochs}'):
                inputs,labels=inputs.to(device),labels.to(device)
                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                loss=self.MSE(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()

            epoch_loss= running_loss / (len(self.train_loader)) 
            # print(f'Epoch {self.epoch+1}')
            print(f'\nTraining Loss : {epoch_loss:.6f}')
            self.training_losses.append(epoch_loss)

            self.validation()

        t_counter += 1
        
    def validation(self):
        self.model.eval()
        val_loss=0.0
        val_MAE = 0.0
        val_MAPE = 0.0
        real_vals = []
        predictions = []

        with torch.no_grad():
            for inputs,labels in tqdm(self.val_loader,desc=f'Calculating Validation loss for Epoch {self.epoch+1}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs=self.model(inputs)
                # print(f'\n\ninputs :\n{inputs}\n\nlabels :\n{labels}\n\nouputs :\n{outputs}\n\n')
                loss=self.MSE(outputs,labels)#MSE
                val_loss+= loss.item()
                mae_loss = self.MAE(outputs,labels)#MAE
                val_MAE+= mae_loss.item()
                MAPE_loss = self.MAPE(outputs,labels)#MAPE
                val_MAPE+= MAPE_loss
                #For R2 calculation
                real_vals.append(labels)
                predictions.append(outputs)

        rea_vals_cum = torch.cat(real_vals)
        predictions_cum = torch.cat(predictions)

        val_R2 =self.R2(predictions_cum,rea_vals_cum).item() #R2
        # print(f'\n\nrea_vals_cum :\n{rea_vals_cum}\n\npredictions_cum :\n{predictions_cum}\n\nval_R2 :\n{val_R2}\n\n')
                
        avg_loss = (val_loss / (len(self.val_loader))) * 100
        print(f'validation loss Avg : {avg_loss:.6f} %\n{"*" * 120}\n')

        self.validation_losses.append(avg_loss)

     
        if avg_loss < self.best_loss:
            print(f'New best model found in epoch: {self.epoch + 1}\n')
            self.best_loss = avg_loss
            model_name = f'{self.best_loss:.6f}.training_{t_counter}_best_model.pth'
            torch.save(self.model.state_dict(), os.path.join(self.best_model_path, model_name))

            log_name = f'{self.best_loss:.6f}.training_{t_counter}_best_model_log.txt'
            with open(os.path.join(self.best_model_path, log_name), 'a') as log_file:
                log_file.write(f'Epoch: {self.epoch + 1}, Validation Loss: {avg_loss:.6f}, MSE: {val_loss}, MAE: {val_MAE}, MAPE: {val_MAPE}, R2: {val_R2}\n')

            self.plot_evaluation(f'{self.best_loss:.6f}.Training and validation losses in {self.epochs} epochs',self.best_loss)

    def plot_evaluation(self, file_name,loss):
        epochs = range(1, len(self.training_losses) + 1)
        # best_val_loss = self.validation_losses[len(self.validation_losses) - 1]

        hyperparameters_label = (f'hidden_size={self.hidden_size}, staked_layers={self.staked_layers}, '
                                f'learning_rate={self.learning_rate}, batch_size={self.batch_size}, '
                                f'epochs={self.epochs}, split={self.split}, loss={loss}')

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.training_losses, label='Training losses', color='blue', marker='o')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)#row , column , number of plot in the figure
        plt.plot(epochs, self.validation_losses, label='Validation losses', color='orange', marker='x')
        plt.title('Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout(rect=[0, 0.01, 1, 0.91])  

        plt.figtext(0.5, -0.02, f'GA - {hyperparameters_label}', ha='center', fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        
        plot_path = os.path.join(sub_directory, f'{file_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        gc.collect()

    def run_model(self):
        self.prepare_data(verbose=False)
        self.training_model()
        # self.plot_evaluation(f'Training and validation losses in {self.epochs} epochs')
        
        return self.best_loss
    
if __name__ == '__main__':
    model = SimpleGA(population_size=10, chromosome_len=17,generations=10,tournament_size=2,crossover_rate=1.0,inherited_rate=0.5,mutation_rate=0.05)
    model.simpleGA(verbose=True,plot=True)
    
