from manim import *
from manim.utils.color import Colors

from mapf import do_algorithm_thingy
from models import MoveType


class MAPFScene(MovingCameraScene):
    def __init__(self, **kwargs):
        MovingCameraScene.__init__(self, **kwargs)
        self.grid, self.solution, self.agents = do_algorithm_thingy()

    @staticmethod
    def node_color(position) -> str:
        # return {self.grid.start: "green", self.grid.end: "red"}.get(position, "#FFFFFF")
        return "#FFFFFF"

    @staticmethod
    def agent_color(agent_idx):
        return ["blue", "yellow", "gold", "teal", "red", "maroon", "green", "purple"][
            agent_idx % 8
            ]

    def construct(self) -> None:
        # self.camera.frame.scale(1.4).move_to((3, 3, 0))
        # self.move_camera((-3, -3))
        # self.camera_frame.set_width(self.camera_frame.get_width() * 1.2)
        # self.scale(1.2)
        dots_indexed = {}
        dots = VGroup()
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                dot = Dot(point=(c, r, 0), color=self.node_color((c, r)))
                dots_indexed[(c, r)] = dot
                dots.add(dot)

        self.play(self.camera.auto_zoom(dots), run_time=0.1)

        self.add(dots)
        self.play(Create(dots), run_time=1)

        if not self.solution:
            self.add(Text("No solution found", color=RED).scale(0.5).shift((2, 1, 0)))

        for (agent_idx, agent) in enumerate(self.agents):
            line_color = self.agent_color(agent.idx)

            start_dot = dots_indexed[(agent.start.x, agent.start.y)]
            end_dot = dots_indexed[(agent.end.x, agent.end.y)]
            # start_dot.set_fill(getattr(Colors, "{}_a".format(line_color)).value)
            start_dot.set_fill(Colors.pure_green.value)
            # end_dot.set_fill(getattr(Colors, "{}_e".format(line_color)).value)
            end_dot.set_fill(Colors.pure_red.value)

            if self.solution:
                solution = self.solution[agent_idx]

                line = VGroup()

                for ((fr, move_fr), (to, move_to)) in zip(solution, solution[1:]):
                    if move_to.mtype == MoveType.NORMAL:
                        line_part_scene = Line(
                            (fr.x, fr.y, 0),
                            (to.x, to.y, 0),
                            stroke_width=8,
                            color=getattr(Colors, line_color).value,
                        )
                    elif move_to.mtype == MoveType.UNDERGROUND:
                        line_part_scene = DashedLine(
                            (fr.x, fr.y, 0),
                            (to.x, to.y, 0),
                            color=getattr(Colors, line_color).value,
                            stroke_width=8,
                            dashed_ratio=0.2,
                        )
                    else:
                        print("ERROR: Unknown move type")
                        return
                    line.add(line_part_scene)
                self.add(line)
                self.play(Create(line), run_time=0.5)
            else:
                line_part_scene = DashedLine(
                    (agent.start.x, agent.start.y, 0),
                    (agent.end.x, agent.end.y, 0),
                    color=getattr(Colors, line_color).value,
                )
                self.add(line_part_scene)
                self.play(Create(line_part_scene), run_time=1)

        self.wait(5)

# class TestScene(Scene):
#     def construct(self):
#         circle = Circle()  # create a circle
#         circle.set_fill(PINK, opacity=0.5)  # set color and transparency
#
#         square = Square()  # create a square
#         square.flip(RIGHT)  # flip horizontally
#         square.rotate(-3 * TAU / 8)  # rotate a certain amount
#
#         self.play(Create(square))  # animate the creation of the square
#         self.play(Transform(square, circle))  # interpolate the square into the circle
#         self.play(FadeOut(square))  # fade out animation
