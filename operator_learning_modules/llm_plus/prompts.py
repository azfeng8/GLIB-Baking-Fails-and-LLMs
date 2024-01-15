STATE_TRANSLATION_PROMPT = """;;;; Translate the following state into natural language.

Q:
clear(b:block), on(b:block,c:block), on(d:block,e:block), ontable(f:block), on(c:block,d:block), handfull(robot:robot), on(e:block,f:block), holding(a:block)

A:
In this scenario, there is a setup involving various blocks and a robot. First, there is a clear block (let's call it 'b') which means it is not being held by the robot or placed on top of any other block. Additionally, block 'b' is on top of another block 'c'. Similarly, there is a block 'd' on top of another block 'e', and a block 'f' is on the table, not stacked on anything. Block 'c' is also on top of block 'd'. The robot's hand is full, meaning it's holding something (let's say block 'a'). Lastly, block 'e' is on top of block 'f'. So, in summary, there are several blocks stacked on top of each other, a block on the table, and the robot is holding one of the blocks.

Q:
on(a:block,b:block), clear(a:block), on(b:block,c:block), on(d:block,e:block), ontable(f:block), on(c:block,d:block), on(e:block,f:block), handempty(robot:robot)

A:
In this state, there are several blocks and a robot involved. Block 'a' is on top of block 'b,' and both of them are clear, meaning there is nothing on top of them. Block 'b' is also on top of block 'c.' Additionally, block 'd' is on top of block 'e,' and block 'f' is on the table, not on top of any other block. Block 'c' is on top of block 'd,' and block 'e' is on top of block 'f.' Lastly, the robot's hand is empty, indicating that the robot is not holding any blocks at the moment.
"""