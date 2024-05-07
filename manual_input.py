# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:31:56 2024

@author: dbromley
"""


import tkinter as tk
from tkinter import filedialog
import csv

class MatrixApp:
    def __init__(self, root, n):
        self.root = root
        self.n = n
        self.matrix = [[' ' for _ in range(n)] for _ in range(n)]
        self.arrows = [[0 for _ in range(n)] for _ in range(n)]
        self.arrow_values = {'↑': 1, '→': 2, '↓': 3, '←': 4, '↷': 5, '↶': 6, 'X': -1} # ↷ = clockwise vortex, ↶ = anticlockwise vortex, X = don't know, could be damage etc
        self.create_widgets()

    def create_widgets(self):
        self.matrix_frame = tk.Frame(self.root)
        self.matrix_frame.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.matrix_frame)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.scrollbar = tk.Scrollbar(self.matrix_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.bind('<Configure>', self.on_canvas_configure)

        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor=tk.NW)

        self.entry_widgets = [[None for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                color = 'white' if (i+j) % 2 == 0 else 'gray'
                entry = tk.Entry(self.frame, width=3, bg=color)
                entry.grid(row=i, column=j)
                entry.bind('<Button-1>', lambda event, row=i, col=j: self.set_current_cell(row, col))
                self.entry_widgets[i][j] = entry

        self.arrow_frame = tk.Frame(self.root)
        self.arrow_frame.pack()

        arrow_directions = ['↑', '→', '↓', '←','↷','↶','X']
        for direction in arrow_directions:
            button = tk.Button(self.arrow_frame, text=direction, command=lambda d=direction: self.set_arrow(d))
            button.pack(side=tk.LEFT)

        self.load_filename_entry = tk.Entry(self.arrow_frame)
        self.load_filename_entry.pack(side=tk.LEFT)
        self.load_filename_entry.insert(0, "filename.csv")

        self.save_filename_entry = tk.Entry(self.arrow_frame)
        self.save_filename_entry.pack(side=tk.LEFT)
        self.save_filename_entry.insert(0, "filename.csv")

        self.save_button = tk.Button(self.arrow_frame, text="Save", command=self.save_matrix)
        self.save_button.pack(side=tk.LEFT)

        self.load_button = tk.Button(self.arrow_frame, text="Load", command=self.load_matrix)
        self.load_button.pack(side=tk.LEFT)

        self.even_direction_var = tk.StringVar(self.arrow_frame)
        self.even_direction_var.set('→')  # Default direction for even rows
        self.even_direction_menu = tk.OptionMenu(self.arrow_frame, self.even_direction_var, *arrow_directions)
        self.even_direction_menu.pack(side=tk.LEFT)

        self.odd_direction_var = tk.StringVar(self.arrow_frame)
        self.odd_direction_var.set('↓')  # Default direction for odd rows
        self.odd_direction_menu = tk.OptionMenu(self.arrow_frame, self.odd_direction_var, *arrow_directions)
        self.odd_direction_menu.pack(side=tk.LEFT)

        self.set_arrows_button = tk.Button(self.arrow_frame, text="Set Arrows", command=self.set_arrows)
        self.set_arrows_button.pack(side=tk.LEFT)

        # Button to check vertices
        self.check_vertices_button = tk.Button(self.arrow_frame, text="Check Vertices", command=self.check_vertices)
        self.check_vertices_button.pack(side=tk.LEFT)

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def set_current_cell(self, row, col):
        self.current_cell = (row, col)

    def set_arrow(self, direction):
        if hasattr(self, 'current_cell'):
            row, col = self.current_cell
            self.arrows[row][col] = self.arrow_values[direction]
            self.update_matrix()

    def set_arrows(self):
        even_direction = self.even_direction_var.get()
        odd_direction = self.odd_direction_var.get()
        for i in range(self.n):
            if i % 2 == 0:  # Even row
                for j in range(self.n):
                    if self.entry_widgets[i][j]['bg'] == 'white':
                        self.arrows[i][j] = self.arrow_values[even_direction]
            else:  # Odd row
                for j in range(self.n):
                    if self.entry_widgets[i][j]['bg'] == 'white':
                        self.arrows[i][j] = self.arrow_values[odd_direction]
        self.update_matrix()

    def update_matrix(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i][j]:
                    self.entry_widgets[i][j].delete(0, tk.END)
                    self.entry_widgets[i][j].insert(0, self.matrix[i][j])
                else:
                    self.entry_widgets[i][j].delete(0, tk.END)

                arrow_value = self.arrows[i][j]
                if arrow_value:
                    for direction, value in self.arrow_values.items():
                        if value == arrow_value:
                            self.entry_widgets[i][j].insert(tk.END, direction)

    def save_matrix(self):
        filename = self.save_filename_entry.get()
        if filename:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(self.n):
                    writer.writerow(self.matrix[i])
                for i in range(self.n):
                    writer.writerow(self.arrows[i])

    def load_matrix(self):
        filename = self.load_filename_entry.get()
        if filename:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
                for i in range(self.n):
                    for j in range(self.n):
                        self.matrix[i][j] = data[i][j]
                for i in range(self.n):
                    for j in range(self.n):
                        self.arrows[i][j] = int(data[self.n + i][j])
                self.update_matrix()


    def check_vertices(self):
        for i in range(1, self.n-1,2): #this needs conditions for that allow for the saturation states, as this dictates where the vertices will be on the checker board
        #i.e. if I initalise <- and ^, the the vertices will show up, but if I do -> and ^ then they don't work, as the places being checked have circular direction of magnets
        #the vertices conditions also need fixing to allow symmetry
            for j in range(1, self.n - 1):
                if self.entry_widgets[i][j]['bg'] != 'white':
                    #these conditions need fixing!!!! (not following the rules properly, and when you meet more than one condition they give double values also does take
                    #into account the square gap between within the lattice)
                    
                    # Type 1 vertex    i= vertical, j= horizontal i.e. i = row num, j = column num, therefore +1 = to the right or down, -1 = to left of above
                    if (
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 3 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 4) or
                        (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 2)
                    ):
                        self.entry_widgets[i][j].configure(bg='green')
                        self.entry_widgets[i][j].insert(tk.END, "V1")
                    
                    # Type 2 vertex
                    if (
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 4) or
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 2) or
                       (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 3 and
                        self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 2) or
                      (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 3 and
                       self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 4)
                    ):
                        self.entry_widgets[i][j].configure(bg='yellow')
                        self.entry_widgets[i][j].insert(tk.END, "V2")
                      
                    # Type 3 vertex
                    if (
                        #1-in 3-out
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 4) or
                        (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 2) or
                        (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 3 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 2) or
                        (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 4)
                        
                        
                    ):
                        self.entry_widgets[i][j].configure(bg='orange')
                        self.entry_widgets[i][j].insert(tk.END, "V3")
                    
                    #3-in 1-out

                    if (
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 2) or
                        (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 3 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 2) or
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 3 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 2) or
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 3 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 4)
                            
                            
                    ):
                        self.entry_widgets[i][j].configure(bg='dodgerblue')
                        self.entry_widgets[i][j].insert(tk.END, "V3")
                    
                    # Type 4 vertex all in
                    if (
                        (self.arrows[i - 1][j] == 3 and self.arrows[i + 1][j] == 1 and
                         self.arrows[i][j - 1] == 2 and self.arrows[i][j + 1] == 4) 
                    ):
                        self.entry_widgets[i][j].configure(bg='red')
                        self.entry_widgets[i][j].insert(tk.END, "V4")
                      
                    # Type 4 vertex all out
                    if (
                        (self.arrows[i - 1][j] == 1 and self.arrows[i + 1][j] == 3 and
                         self.arrows[i][j - 1] == 4 and self.arrows[i][j + 1] == 2) 
                    ):
                        self.entry_widgets[i][j].configure(bg='purple')
                        self.entry_widgets[i][j].insert(tk.END, "V4")
                    '''   
                    #need a condtion where if none of these are met i.e. there are vortices etc, then it gives nothing
                    else:
                        self.entry_widgets[i][j].configure(bg='pink')
                        self.entry_widgets[i][j].insert(tk.END, "??")
                    '''

def main():
    root = tk.Tk()
    root.title("ASI Matrix")
    num_bars = 5  # Size of the chessboard
    n = num_bars * 2
    app = MatrixApp(root, n)
    root.mainloop()

if __name__ == "__main__":
    main()
