import time
import sys
 
import pygame
from pygame.locals import *

# SPACING
PAD = 10

# COLORS
BLACK = pygame.Color((0,0,0))
WHITE = pygame.Color((255,255,255))
RED  = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
DARK_BG = pygame.Color('#2f3e46')
DARK_BG_OUTLINE = pygame.Color('#354f52')
MID = DARK_BG.lerp(DARK_BG_OUTLINE, 0.5)

class Window:
  def __init__(self, top, left, w=100, h=100, bg=BLACK, outline_width=0, outline_color=WHITE, parent=None): 
    self.parent = parent
    self.top = top
    self.left = left
    self.w = w
    self.h = h
    self.bg = bg
    self.surf = pygame.Surface((w,h))
    self.surf.fill(self.bg)
    self.outline_width = outline_width
    self.outline_color = outline_color
    self.children = []

  @property
  def bot(self):
    return self.top + self.h
  
  def draw(self, surf):
    self.surf.fill(self.bg)
    self.outline()
    #surf.blit(self.surf, (self.left, self.top))
    for c in self.children:
      c.draw(self.surf)

    surf.blit(self.surf, (self.left, self.top))

  def outline(self):
    ow = self.outline_width
    how = ow / 2
    fixeven = -1 if ow % 2 == 0 else 0
    offs = int(how)
    pygame.draw.line(self.surf, self.outline_color, (0, offs + fixeven), (self.w, offs + fixeven), ow) # top
    pygame.draw.line(self.surf, self.outline_color, (0,self.h - (offs+1)), (self.w, self.h - (offs+1)), ow) # bot
    pygame.draw.line(self.surf, self.outline_color, (offs,0), (offs, self.h), ow) # left 
    pygame.draw.line(self.surf, self.outline_color, (self.w - (offs + 1 + fixeven),0), (self.w - (offs + 1 + fixeven), self.h), ow) # right

  def add_child(self, window):
    self.children.append(window)

class GraphWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def draw(self, surf):
    super().draw(surf)
    n = game.recorder.n
    prev = None
    for i, sample in enumerate(game.recorder.samples):
      color = DARK_BG.lerp(WHITE, .5)
      cx = self.left + (i/(n-1))*self.w
      cy = self.bot - sample*self.h
      pygame.draw.circle(surf, color, (cx,cy), 2)
      if i > 0:
        pygame.draw.aaline(surf, color, prev_point, (cx,cy),1)
      prev_point = (cx, cy)

class HoverWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.hovered = False

  def draw(self, surf):
    super().draw(surf)
    if game.pointer.hovering:
      self.draw_cursor(surf, game.pointer.xy[0])

  def draw_cursor(self, surf, x):
      color = DARK_BG_OUTLINE
      width = 3
      pygame.draw.line(surf, color, (x,self.top), (x, self.top + self.h), width)
      pygame.draw.line(surf, color.lerp(DARK_BG, 0.6), (x + width,self.top), (x + width, self.top + self.h), width)
      pygame.draw.line(surf, color.lerp(DARK_BG, 0.6), (x - width,self.top), (x - width, self.top + self.h), width)

class Pointer:
  hover = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
  normal = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_ARROW)
  hand = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_HAND)

  def __init__(self, game):
    self.game = game
    self.icon = Pointer.hover

    self.hover_zone_left = self.game.hover_zone.left
    self.hover_zone_right = self.game.hover_zone.left + self.game.hover_zone.w
    self.hover_zone_top = self.game.hover_zone.top
    self.hover_zone_bot = self.game.hover_zone.top + self.game.hover_zone.h
    self.hover_val = None # 0?
    self.hovering = False


  def in_hover_zone(self):
    x, y = self.xy
    if y < self.hover_zone_top or y > self.hover_zone_bot or x < self.hover_zone_left or x > self.hover_zone_right:
      self.hovering = False
      return False
    T = x - self.hover_zone_left
    self.hover_val = rescale(T, mn=0, mx=self.game.hover_zone.w, a=0, b=1)
    self.hovering = True
    return True

  def report(self):
    self.xy = pygame.mouse.get_pos()
    if self.xy[0] > self.game.W/2:
      pygame.mouse.set_cursor(Pointer.hover)
    elif self.in_hover_zone():
      pygame.mouse.set_cursor(Pointer.hand)
#      self.game.hover_zone.draw_cursor(self.xy[0])
    else:
      pygame.mouse.set_cursor(Pointer.normal)

### utility funcs ###

def clamp(t, mn=0, mx=1):
  return max(mn, min(mx, t))

def rescale(t, mn, mx, a=0, b=1):
  return a + (b-a)*(t-mn)/(mx-mn)

def lerp(t, a, b):
  return a + t*(b-a)

### end utility funcs ###

class Game:

  def __init__(self):
    pygame.init()

    self.W = 640*2
    self.H = 480
    self.outline_width = 3

    self.FPS = 60.0
    self.clock = pygame.time.Clock()


    self.screen = pygame.display.set_mode((self.W, self.H))

    left_window = Window(top = 0, left=0, w=self.W//2, h=self.H, bg=DARK_BG, outline_width=self.outline_width, outline_color=DARK_BG_OUTLINE)

    hover_w, hover_h = left_window.w *0.8, left_window.h*0.2
    self.hover_zone = HoverWindow(top = left_window.h//2 - hover_h//2, left = (left_window.w - hover_w)//2, w=hover_w, h=hover_h,
                                  bg=MID, outline_width=1, outline_color=DARK_BG_OUTLINE ) 
    left_window.add_child(self.hover_zone)

    graph_w, graph_h = left_window.w *0.8, (left_window.h - self.hover_zone.bot - PAD*2)
    self.graph_zone = GraphWindow(top = self.hover_zone.bot + PAD, left = self.hover_zone.left, w=graph_w, h=graph_h,
                                  bg=MID, outline_width=1, outline_color=DARK_BG_OUTLINE ) 
    left_window.add_child(self.graph_zone)

    right_window = Window(top = 0, left=self.W//2, w=self.W//2, h=self.H, bg=DARK_BG, outline_width=self.outline_width, outline_color=DARK_BG_OUTLINE)

    self.windows = [left_window, right_window]

    self.pointer = Pointer(self)
    self.recorder = Recorder(self)

  def draw(self):
    self.screen.fill(BLACK)

    for win in self.windows:
      win.draw(self.screen)

    pygame.display.flip()

  def update(self):
    dt = self.dt
    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()
        sys.exit() 
      elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_r:
            game.recorder.start()

    self.pointer.report()
    self.recorder.update(dt)

  def run(self):
    self.dt = 1/self.FPS
    while True:
      self.update()
      self.draw()
      self.dt = self.clock.tick(self.FPS)

class Recorder:
  def __init__(self, game):
    self.game = game
    self.sr = 24 # samples per second
    self.dur = 3 # capture seconds
    self.T = 1/self.sr # period in seconds
    self.n = int(self.sr*self.dur)
    self.samples = []
    self.live = False


  def update(self, dt): # dt in ms
    if self.live:
      t = time.time()
      elapsed = t-self.t0
      if elapsed >= self.T*self.i:
        self.sample()

  def sample(self):
    #print('sample taken')
    T = self.game.pointer.hover_val
    self.samples.append(T)
    #self.game.graph_zone.mark_sample(self.i, self.n)
    self.i += 1
    if self.i == self.n: # finished sampling 
      self.live = False
      total_elapsed = time.time() - self.t0
      print(self.samples)
      print(f'finished: took {self.n} samples in {total_elapsed} seconds.') 

  def start(self):
    self.live = True
    self.i = 0
    self.samples = []
    self.t0 = time.time()
    self.sample()
    print('started recording')

game = Game()
game.run()