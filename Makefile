base:
	python grid.py
visualize:
	manim -p -ql visualize.py MAPFScene
do_profile:
	kernprof -l -v grid.py
read_profile:
	python -m line_profiler grid.py.lprof
