base:
	python mapf.py
visualize:
	manim -p -ql visualize.py MAPFScene
visualize-high:
	manim -p visualize.py MAPFScene

do_profile:
	kernprof -l -v mapf.py
read_profile:
	python -m line_profiler grid.py.lprof
