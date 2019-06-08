import board_conversions as bc
from multiprocessing import Pool, Queue
import threading
import asyncio
import random
import requests
import json

class GetSolnThread(threading.Thread):
	def __init__(self, t, board='na'):
		if board == 'na':
			board = bc.generate_random_board_state()
		threading.Thread.__init__(self, target=t, kwargs = {'board' : board})
		self.start()

class ConnectFourDataGenerator:

	def __init__(self, num_threads=10, boards=[]):
		self.num_threads = num_threads
		self.row_data = []
		self.lock = threading.Lock()
		if boards == []:
			for i in range(num_threads):
				boards.append(bc.generate_random_board_state())
		self.boards = boards

	def _generate_soln(self, board):
		new_row = self.get_soln(board)
		if new_row == 'nosoln':
			return
		self.lock.acquire()
		self.row_data.append(new_row)
		self.lock.release()

	def get_soln(self, board_state, return_pos=True):
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
		return 'nosoln'

	def gather_data(self):
		all_threads = []
		for i in range(self.num_threads):
			try:
				thread = GetSolnThread(self._generate_soln, self.boards[i])
				all_threads.append(thread)
			except Exception as e:
				print(e)
				print("Thread not started")

		for thread in all_threads:
			thread.join()

		return self.row_data
