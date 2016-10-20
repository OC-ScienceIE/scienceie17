#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
'''
This script reads Semeval ScienceIE 2017 annotation files (.ann) in directory _textfolder_ and obtains some basic statistics 
about entities (task, material, process), relations and embeddings. 
By uncommenting the print statements after line 214, it is possible to print embeddings' instances. 

@author: cristism
@version 2016-10-19
'''

import os, re, string


def characters(entity):
    characters = []
    for line in entity:
        characters.append(len(line[2])) 
    return characters


def longest_entity(entity,characters):
    longest = []
    for line in entity:
        if len(line[2]) == max(characters):
            longest.append(line[2])
    return longest  

def shortest_entity(entity,characters):
    shortest = []
    for line in entity:
        if len(line[2]) == min(characters):
            shortest.append(line[2])
    return shortest
    
def num_capitals(entity):
    capital=[]
    for line in entity:
        capital.append(len(filter(lambda x: x in string.uppercase, line[2])))
    return sum(capital)

def num_words(entity):
    words = []   
    for line in entity:
        word1 = line[2].split()
        words.append(len(word1))
    return sum(words)

def all_capitals(entity):
    capitals =[]
    for line in entity:
        if line[2].isupper():
            capitals.append(line[2])
    return capitals



textfolder = "/Users/Cristina/Documents/Conferences/20170303_Semeval/scienceie/train"
#textfolder = "/Users/Cristina/Documents/Conferences/20170303_Semeval/scienceie/train"
# error corrected in training file S0167931713005042.ann:T6	Task 400 436;437 453	Nanofeature stamps were then created from the samples

# list of ann files in directory
flist = []
for file in os.listdir(textfolder):
    if file.endswith(".ann"):
        flist.append(file)


alle = []# list of lists including filenames and their lines, like this 
# [['T6', 'Task 192 219', 'novel algorithm is proposed'], 'S2212671612002181.ann']
# [['T7', 'Process 223 254', 'optimize the hidden node number'], 'S2212671612002181.ann']

material = []# list of lists with materials, no filenames, e.g ['T22', 'Material 1142 1149', 'tritium'], ['T23', 'Material 1188 1210', 'carbon based materials'], ['T4', 'Material 0 26', 'Power and particle exhaust']]
process = [] # list of lists with processes, no filenames
task = []  # list of lists with tasks, no filenames
hyponym = [] # list of lists with hyponyms, no filenames
synonym = [] # list of lists with synonyms, no filenames


for f in flist: 
    f_anno = open(os.path.join(textfolder, f), "r")
    for line in f_anno:
        #alle.append(line)
        #print line
        anno_inst = line.strip().split("\t")
        alle.append([anno_inst,f])
        # MATERIAL
        if re.match(r"Material",anno_inst[1]):
            material.append(anno_inst)
        # PROCESS
        elif re.match(r"Process",anno_inst[1]):
            process.append(anno_inst)
        # TASK
        elif re.match(r"Task",anno_inst[1]):
            task.append(anno_inst)   
        # HYPONYM
        elif re.match(r"Hyponym",anno_inst[1]):
            hyponym.append(anno_inst)
        # SYNONYM
        elif re.match(r"Synonym",anno_inst[1]):
            synonym.append(anno_inst)


# EMBEDDINGS
from collections import defaultdict

# create dictionary of files and their annotations of entities, ordered by the first line number
embeddings_entities = defaultdict(list)
# S0022311514006722.ann [['Material', '0', '16'], ['Material', '76', '92'], ['Process', '191', '213'], ['Process', '222', '231'], ['Process', '279', '288'], ['Process', '308', '324'], ['Material', '362', '366'], ['Process', '394', '403'], ['Material', '422', '438'], ['Material', '627', '632'], ['Task', '694', '744'], ['Process', '735', '744'], ['Task', '757', '825'], ['Process', '808', '825']]
relations = defaultdict(list)


for line in alle:
 #   [['T6', 'Task 615 639', 'plasma facing components'], 'S2352179114200056.ann']
    entities = line[0][1].split()
    line[0][1] = line[0][1].split()
    if entities[0] == "Material" or entities[0] == "Process" or entities[0] == "Task":
        # embeddings_entities[line[1]].append(entities)
        embeddings_entities[line[1]].append(line[0])
    else:
        relations[line[1]].append(entities)


    
freq_relations = []
for f,entities in relations.items(): 
    #print f,entities
    #S0370269304009657.ann [['Hyponym-of', 'Arg1:T3', 'Arg2:T4'], ['Synonym-of', 'T10', 'T9']]
    freq_relations.append(len(entities))


# order list of files and their entities by first character number within each file
freq_entities = []
# create list of the entities strings
list_strings =[]
for f,entities in embeddings_entities.items(): 
    #print f,entities
    #S0021999114008523.ann [['T1', ['Task', '1212', '1279'], 'multiscale systems comprising an arbitrary number of coupled models'], ['T2', ['Process', '2', '50'], 'multi-physics description of a multiscale system'], ['T3', ['Process', '77', '91'], '\xe2\x80\x98hybrid\xe2\x80\x99 model'], ['T4', ['Process', '122', '128'], 'hybrid'], ['T5', ['Process', '140', '159'], 'molecular treatment'], ['T6', ['Process', '163', '176'], '\xe2\x80\x98micro\xe2\x80\x99 model'], ['T7', ['Process', '185', '204'], 'continuum-fluid one'], ['T8', ['Process', '209', '221'], 'macro\xe2\x80\x99 model'], ['T9', ['Process', '322', '344'], 'micro and macro models'], ['T10', ['Process', '953', '993'], 'time-stepping method for coupled systems'], ['T11', ['Process', '1057', '1080'], 'continuous asynchronous'], ['T12', ['Process', '1082', '1084'], 'CA'], ['T13', ['Process', '1117', '1139'], 'micro and macro models'], ['T14', ['Process', '668', '683'], 'scale-separated'], ['T15', ['Process', '688', '739'], 'physical (as distinct from numerical) approximation'], ['T16', ['Material', '96', '101'], 'fluid'], ['T17', ['Task', '240', '310'], 'obtaining the accuracy of the former with the efficiency of the latter'], ['T18', ['Task', '96', '110'], 'fluid dynamics'], ['T19', ['Process', '769', '783'], 'coupled models'], ['T20', ['Process', '1265', '1279'], 'coupled models']]    
    entities.sort(key=lambda x: int(x[1][1]))
    freq_entities.append(len(entities))
    for e in entities:
        list_strings.append(e[2])



## GET DICTIONARY OF STRINGS OF ENTITIES

# Given a list of words, return a dictionary of
# word-frequency pairs.
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))

# Sort a dictionary of word-frequency pairs in
# order of descending frequency.

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


dict_freq_entities = wordListToFreqDict(list_strings)
ordered_dict = sortFreqDict(dict_freq_entities)

outf_dict = open("freq_dict.txt", "w")
for freq,ent in ordered_dict:
    print freq,ent
    outf_dict.write(str(freq) + "\t" + str(ent) + "\n")

outf_dict.close()

# BASIC STATISTICS ABOUT FILES
print "................................................"
print "................................................"

print "BASIC STATISTICS"
print "Total number of files: ", len(flist)
print "Total number of entities and relations: ", len(alle)
print "TOTAL NUMBER OF ENTITIES (MATERIAL, PROCESS, TASK)", sum(freq_entities)
print "TOTAL NUMBER OF RELATIONS (SYNONYM, HYPONYM)", sum(freq_relations)
print ".........................."
print ".........................."
# BASIC STATISTICS ABOUT CLASSES  
print "BASIC STATISTICS ABOUT ENTITIES"
print ".........................."
print "MATERIAL"
print "Total number of MATERIAL: ", len(material)
print "Average number of MATERIAL per document: ", float(len(material))/float(len(flist))


ch_material = characters(material)
print "Average length of MATERIAL (in characters): ", float(sum(ch_material)) / float(len(material))
print "Average length of MATERIAL (in words): ", float(num_words(material)) / float(len(material))
print "Longest MATERIAL: ", longest_entity(material,ch_material)
print "Shortest MATERIAL: ", shortest_entity(material,ch_material)
print "Average number of words with capital letters in MATERIAL: ", float(num_capitals(material)) / float(num_words(material))
print "Average number of MATERIALS written only in capital letters, as f.e. in acronyms: ", float(len(all_capitals(material))) / float(len(material))


print ".........................."

print "Total number of PROCESS: ", len(process)
print "Average number of PROCESS per document: ", float(len(process))/float(len(flist))
ch_process = characters(process)
print "Average length of PROCESS (in characters): ", float(sum(ch_process)) / float(len(process))
print "Average length of PROCESS (in words): ", float(num_words(process)) / float(len(process))
print "Longest PROCESS: ", longest_entity(process,ch_process)
print "Shortest PROCESS: ", shortest_entity(process,ch_process)
print "Average number of words with capital letters in PROCESS: ", float(num_capitals(process)) / float(num_words(process))
print "Average number of PROCESSS written only in capital letters, as f.e. in acronyms: ", float(len(all_capitals(process))) / float(len(process))
print ".........................."


print "Total number of TASK: ", len(task)
print "Average number of TASK per document: ", float(len(task))/float(len(flist))
ch_task = characters(task)
print "Average length of TASK (in characters): ", float(sum(ch_task)) / float(len(task))
print "Average length of TASK (in words): ", float(num_words(task)) / float(len(task))
print "Longest TASK: ", longest_entity(task,ch_task)
print "Shortest TASK: ", shortest_entity(task,ch_task)
print "Average number of words with capital letters in TASK: ", float(num_capitals(task)) / float(num_words(task))
print "Average number of TASKS written only in capital letters, as f.e. in acronyms: ", float(len(all_capitals(task))) / float(len(task))

print ".........................."
print ".........................."  
# BASIC STATISTICS ABOUT CLASSES  
print "BASIC STATISTICS ABOUT RELATIONS"
print ".........................."
    
# BASIC STATISTICS ABOUT RELATIONS
print "Total number of HYPONYM: ", len(hyponym)
print "Average number of HYPONYM per document: ", float(len(hyponym))/float(len(flist))
print "Total number of SYNONYM: ", len(synonym)
print "Average number of SYNONYM per document: ", float(len(synonym))/float(len(flist))
print ".........................."



##### EMBEDDINGS     

print "................................................"
print "................................................"
print "................................................"
print "................................................"
print "INSTANCES AND BASIC STATISTICS ABOUT EMBEDDINGS"

task_in_process =[]
task_in_material =[]
task_in_task =[]
process_in_task =[]
process_in_material =[]
process_in_process =[]
material_in_task =[]
material_in_process =[]
material_in_material =[]
embeddings = []
d_task_task ={}

for f,entities in embeddings_entities.items():
    #print f,entities 
    for index, elem in enumerate(entities):
       # print index, elem
        #0 ['T2', ['Process', '2', '50'], 'multi-physics description of a multiscale system']
        #1 ['T3', ['Process', '77', '91'], '\xe2\x80\x98hybrid\xe2\x80\x99 model']
        if index > 0:
#             print entities[index][1],entities[index-1][2]
            if int(entities[index][1][1]) < int(entities[index-1][1][2]):
                embeddings.append(1)
                if len(entities[index-1][2]) > len(entities[index][2]):              
                    #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""                    
                    if entities[index-1][1][0] == "Material":
                        if entities[index][1][0] == "Task":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""                            
                            task_in_material.append(1)
                        elif entities[index][1][0] == "Process":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""                            
                            process_in_material.append(1)
                        elif entities[index][1][0] == "Material":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""
                            material_in_material.append(1)
                    elif entities[index-1][1][0] == "Process":
                        if entities[index][1][0] == "Task":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""
                            task_in_process.append(1)
                        elif entities[index][1][0] == "Process":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""                            
                            process_in_process.append(1)
                        elif entities[index][1][0] == "Material":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""
                            
                            material_in_process.append(1)
                    elif entities[index-1][1][0] == "Task":
                        if entities[index][1][0] == "Task":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""
                            task_in_task.append(1)
                        elif entities[index][1][0] == "Process":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""
                            process_in_task.append(1)
                        elif entities[index][1][0] == "Material":
                            #print "A", entities[index][1][0], "is inside a", entities[index-1][1][0], "in file", f, ". Indexes are: ", entities[index][1], entities[index-1][1] , "Example \"", entities[index][2], "\" and \"", entities[index-1][2], "\""
                            material_in_task.append(1)           
                else:
                    #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                    
                    if entities[index][1][0] == "Task":
                        if entities[index-1][1][0] == "Process":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            process_in_task.append(1)                            
                        elif entities[index-1][1][0] == "Task":                            
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            task_in_task.append(1)
                        elif entities[index-1][1][0] == "Material":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            material_in_task.append(1) 
                    elif entities[index][1][0] == "Material":
                        if entities[index-1][1][0] == "Process":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            process_in_material.append(1)                            
                        elif entities[index-1][1][0] == "Task":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            task_in_material.append(1)
                        elif entities[index-1][1][0] == "Material":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            material_in_material.append(1)
                    elif entities[index][1][0] == "Process":
                        if entities[index-1][1][0] == "Process":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            process_in_process.append(1)                            
                        elif entities[index-1][1][0] == "Task":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            task_in_process.append(1)
                        elif entities[index-1][1][0] == "Material":
                            #print "A", entities[index-1][1][0], "is inside a", entities[index][1][0], "in file", f, ". Indexes are: ", entities[index-1][1], entities[index][1] , "Example \"", entities[index-1][2], "\" and \"", entities[index][2], "\""
                            
                            material_in_process.append(1)


print "................................................"
print "................................................"
print "Total number of EMBEDDINGS:", len(embeddings)
print "Total number of TASK in PROCESS:", len(task_in_process)   
print "Total number of TASK in MATERIAL:", len(task_in_material)   
print "Total number of TASK in TASK:", len(task_in_task)   
print "Total number of MATERIAL in PROCESS:", len(material_in_process)   
print "Total number of MATERIAL in MATERIAL:", len(material_in_material)   
print "Total number of MATERIAL in TASK:", len(material_in_task)   
print "Total number of PROCESS in PROCESS:", len(process_in_process)   
print "Total number of PROCESS in MATERIAL:", len(process_in_material)   
print "Total number of PROCESS in TASK:", len(process_in_task)   


print "................................................"
print "................................................"


