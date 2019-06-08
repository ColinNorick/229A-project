import requests
from bs4 import BeautifulSoup
import json
import csv
import random
import numpy as np
import board_conversions as bc
import time
from ConnectFourDataGenerator import ConnectFourDataGenerator

def get_soln(board_state, return_pos=True):
		"""
		Gets the solution for a given board state.

		:board_state: A zero-indexed string representing the board state of the C4 game.
		:returns: An length-7 array consisting of the scores of dropping in each of the seven positions.
		when return_pos is True, will append the string of the board state to the solution returned.
		"""
		# re-index the board_state to one-indexed
		#board_state = ''.join([ str(int(ch)+1) for ch in board_state ])

		BASE_URL = "https://connect4.gamesolver.org/solve"
		url = BASE_URL + f"?pos={board_state}"

		request = requests.get(url,
			headers={
				"Accept": "application/json, text/javascript, */*; q=0.01",
				"DNT": "1",
				"Referer": "https://connect4.gamesolver.org/",
				"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
				"X-Requested-With": "XMLHttpRequest"
			},
			cookies={},
		)
		if request.ok:
			content = str(request.content, encoding='utf8')
			soln = json.loads(content)['score']

			# change parity based on the next player
			if len(board_state) % 2 == 1:
				soln = [ -rank for rank in soln ]

			if return_pos:
				pos = json.loads(content)['pos']
				return [pos] + soln
			else:
				return soln

		else:
			request.raise_for_status()

def get_examples(filename="generated-examples.csv", num=1000, verbose=True, overwrite=False):
	open_param = 'a'
	if overwrite:
		open_param = 'w'
	rows = []
	num_threads = 40
	for i in range(num // num_threads):
		if verbose:
			j = num_threads * i
			if j % (num // 100) == 0:
				print('{} % done'.format(j / (num // 100)))

		worker = ConnectFourDataGenerator(num_threads)
		row_data = worker.gather_data()
		rows += row_data

	with open(filename, open_param) as f:
		writer = csv.writer(f, lineterminator='\n')
		writer.writerows(rows)

def get_ply_examples(filename='na.csv', ply=2, num_threads=40, verbose=True, overwrite=True):
	if filename == "na.csv":
		filename = 'exhaustive-' + str(ply) + 'ply.csv'
	open_param = 'a'
	rows = []
	boards = bc.get_all_boards_le(ply)
	num = len(boards)
	print('{} boards at {}-ply'.format(num, ply))
	for i in range(num // num_threads):
		if verbose:
			j = num_threads * i
			if j % (num // 100) == 0:
				print('{} %  done'.format(j / (num // 100)))
				with open(filename, open_param) as f:
					writer = csv.writer(f, lineterminator='\n')
					writer.writerows(rows)
					rows = []

		worker = ConnectFourDataGenerator(num_threads, boards[i:i+num_threads])
		row_data = worker.gather_data()
		rows += row_data
	with open(filename, open_param) as f:
		writer = csv.writer(f, lineterminator='\n')
		writer.writerows(rows)
		rows = []
	




def sanitize_files(filenames, output='examples.csv'):
	total_data = set()

	for filename in filenames:
		with open(filename, 'r') as f:
			lines = f.readlines()

		total_data |= set(lines)

	str_to_write = ''.join(list(total_data))

	with open(output, 'w') as f:
		f.write(str_to_write)

def get_block_examples(filename_base, num_blocks, num_per_block):
	for i in range(num_blocks):
		filename = filename_base + str(i) + '.csv'

		print(f"Writing to {filename}...")
		get_examples(filename=filename, num=num_per_block, overwrite=True)
		print('-'*20)
		print()
	sanitize_files([filename_base + str(i) + '.csv' for i in range(num_blocks)], filename_base + '-ALL.csv')




if __name__ == '__main__':
	DEFAULT_TOTAL_SAMPLES = 100000
	DEFAULT_NUM_FILESPLITS = 10
	DEFAULT_FILENAME_BASE = "bulk_one"

	# print("Welcome to Antonio, Parth, and Colin's Magical Connect 4 AI Adventure.")
	# print("This file generates random Connect 4 AI data by querying")
	# print("https://connect4.gamesolver.org/.")
	# total_samples = input("How many samples would you like to generate? Leave blank for the default: ")
	# filename_base = input("We'll store those in five files, with a base filename. What would you like that to be? Leave blank for the default: ")

	# if total_samples:
	# 	total_samples = int(total_samples)
	# else:
	total_samples = DEFAULT_TOTAL_SAMPLES

	# if not filename_base:
	filename_base = DEFAULT_FILENAME_BASE

	# get_block_examples(filename_base, DEFAULT_NUM_FILESPLITS, total_samples // DEFAULT_NUM_FILESPLITS)
	get_block_examples('test-night', 5, 100)
	print("TEST ALL DONE!!!!!")
	get_block_examples('bulk-two', 10, 10000)
	print("FIRST BLOCK ALL DONE!!!!!")
	get_ply_examples(ply=7)
	print("PLY SEVEN ALL DONE!!!!!")
	get_block_examples('bulk-three', 10, 10000)
	print("SECOND BLOCK ALL DONE!!!!!")