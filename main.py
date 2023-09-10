import time
import sys
import math
from collections import deque, defaultdict
 
import numpy as np
from fit_curve import piecewise3poly
import pygame
from pygame.locals import *

# SPACING
PAD = 10

# COLORS
BLACK = pygame.Color((0,0,0))
WHITE = pygame.Color((255,255,255))
RED  = pygame.Color((255,0,0))
BLUE = (0,0,255)
GREEN = (0,255,0)
DARK_BG = pygame.Color('#2f3e46')
DARK_BG_OUTLINE = pygame.Color('#354f52')
DARK_BTN = pygame.Color('#354f52').lerp(BLACK, 0.3)
DARK_BTN_OUTLINE = pygame.Color('#354f52').lerp(BLACK, 0.4)
MID = DARK_BG.lerp(DARK_BG_OUTLINE, 0.5)
AMBER  = pygame.Color('#828a5b')
BEIGE  = pygame.Color('#f2e3bc')

class Rect(object):
  def __init__(self, top, left, w, h, bg=BLACK, outline_width=1, outline_color=WHITE, parent=None):
    self.parent = parent
    self.top = top
    self.left = left
    self.w = w
    self.h = h
    self.bg = bg
    self._bg = bg
    self.surf = pygame.Surface((w,h))
    self.surf.fill(self.bg)
    self.outline_width = outline_width
    self._outline_width = outline_width
    self.outline_color = outline_color
    self._outline_color = outline_color

  @property
  def bot(self):
    return self.top + self.h - 1
  @property
  def right(self):
    return self.left + self.w - 1
  @property
  def absleft(self):
    if self.parent is None:
      return self.left
    return self.parent.absleft + self.left
  @property
  def absright(self):
    if self.parent is None:
      return self.right
    return self.parent.absleft + self.w
  @property
  def abstop(self):
    if self.parent is None:
      return self.top
    return self.parent.abstop + self.top

  def outline(self):
    ow = self.outline_width
    how = ow / 2
    fixeven = -1 if ow % 2 == 0 else 0
    offs = int(how)
    pygame.draw.line(self.surf, self.outline_color, (0, offs + fixeven), (self.w, offs + fixeven), ow) # top
    pygame.draw.line(self.surf, self.outline_color, (0,self.h - (offs+1)), (self.w, self.h - (offs+1)), ow) # bot
    pygame.draw.line(self.surf, self.outline_color, (offs,0), (offs, self.h), ow) # left 
    pygame.draw.line(self.surf, self.outline_color, (self.w - (offs + 1 + fixeven),0), (self.w - (offs + 1 + fixeven), self.h), ow) # right

### Window class hierarchy ###
class Window(Rect):
  def __init__(self, game, top, left, w=100, h=100, bg=BLACK, outline_width=0, outline_color=WHITE, parent=None): 
    super().__init__(top,left,w,h,bg,outline_width,outline_color,parent)
    self.game = game
    self.children = []

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
      color = DARK_BG.lerp(WHITE, .3)
      cx = self.left + (i/(n-1))*self.w 
      cy = self.bot - 1 - sample*(self.h-3)
      #pygame.draw.circle(surf, color, (cx,cy), 1)
      if i > 0:
        pygame.draw.aaline(surf, color, prev_point, (cx,cy),1)
      prev_point = (cx, cy)

    # draw smoothed fit curve 
    smoothed = game.recorder.smoothed
    n = len(smoothed)
    color = DARK_BG.lerp(WHITE, .5)
    for i, sample in enumerate(smoothed):
      cx = self.left + (i/(n-1))*self.w
      cy = self.bot - 1 - sample*(self.h-3)
      #pygame.draw.circle(surf, color, (cx,cy), 1)
      if i > 0:
        lw = 3
        #pygame.draw.line(surf, color, prev_point, (cx,cy),lw)
        #pygame.draw.aaline(surf, color, prev_point, (cx,cy))
        pygame.draw.aaline(surf, color, prev_point, (cx,cy))
      prev_point = (cx, cy)

    if game.recorder.playing:
      self.draw_cursor(surf, self.left + game.recorder.playback_progress*self.w)

  def draw_cursor(self, surf, x):
      color = DARK_BG_OUTLINE
      width = 2
      pygame.draw.line(surf, color, (x,self.top), (x, self.top + self.h), width)
      pygame.draw.line(surf, color.lerp(DARK_BG, 0.6), (x + width,self.top), (x + width, self.top + self.h), width)
      pygame.draw.line(surf, color.lerp(DARK_BG, 0.6), (x - width,self.top), (x - width, self.top + self.h), width)

class HoverWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.hovered = False
    self.cursor_color = AMBER.lerp(BEIGE, .1).lerp(DARK_BG,.2)
    self.cursor_outline_color = self.cursor_color.lerp(BLACK, 0.3)

  def draw(self, surf):
    super().draw(surf)
    if game.pointer.hovering:
      self.draw_cursor(surf, game.pointer.xy[0])

  def draw_cursor(self, surf, x):
      width = 3
      pygame.draw.line(surf, self.cursor_color, (x,self.top), (x, self.top + self.h), width)
      pygame.draw.line(surf, self.cursor_outline_color, (x + width,self.top), (x + width, self.top + self.h), width)
      pygame.draw.line(surf, self.cursor_outline_color, (x - width,self.top), (x - width, self.top + self.h), width)

class LevelWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sprites = []
    ball_color = RED.lerp(DARK_BG, .5)
    self.ball = Ball(self, fulcrum=(self.w//2, self.h), arm_radius=self.w//4, ball_radius=12, phase=0.5, color=ball_color, group=self.sprites)

  def update(self, dt):
    for sprite in self.sprites:
      sprite.update(dt)

  def draw(self, surf):
    self.surf.fill(self.bg)
    self.outline()

    for sprite in self.sprites:
      sprite.draw(self.surf)

    surf.blit(self.surf, (self.left, self.top))

class ControlPanelWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    bPAD = PAD//2
    button_w = self.w - 2*bPAD
    a = ControlPanelButton(top=bPAD, left=bPAD, w=button_w, h = button_w, parent=self, id=0 )
    b = ControlPanelButton(top=bPAD + (button_w + bPAD), left=bPAD, w=button_w, h = button_w, parent=self, id=1)
    c = ControlPanelButton(top=bPAD + 2*(button_w + bPAD), left=bPAD, w=button_w, h = button_w, parent=self, id=2 )
    d = ControlPanelButton(top=bPAD + 3*(button_w + bPAD), left=bPAD, w=button_w, h = button_w, parent=self, id=3 )
    a.click()

  def draw(self, surf):
    self.surf.fill(self.bg)
    self.outline()

    for button in ControlPanelButton.group:
      button.draw(self.surf)

    surf.blit(self.surf, (self.left, self.top))


class ListenerPanelWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    bhPAD = PAD//2
    button_w = self.w - 2*bhPAD
    n_buttons = 4 # TODO: pull this value from LevelConfig object
    v_heights = get_spacing(total=self.h, s=button_w, n=n_buttons)
    for i in range(n_buttons):
      ListenerPanelButton(top=v_heights[i], left=bhPAD, w=button_w, h=button_w, parent=self,id=i)

  def draw(self, surf):
    self.surf.fill(self.bg)
    self.outline()

    for button in ListenerPanelButton.group:
      button.draw(self.surf)

    surf.blit(self.surf, (self.left, self.top))

class DebugWindow(Window):
  MSGBOX_OFFSET = (10,10)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.font = pygame.font.SysFont("monospace", 18)
    self.messages = []
    self.label = ""
    self.color = BLACK

  def print(self, text):
    self.messages.append(str(text))

  def update(self, dt):
    text = "\n".join(self.messages)
    self.label = self.font.render(text, 1, self.color)
    self.messages = []

  def draw(self, surf):
    self.surf.fill(self.bg)
    self.outline()
    self.surf.blit(self.label, DebugWindow.MSGBOX_OFFSET)

    surf.blit(self.surf, (self.left, self.top))

### end Window class hierarchy ###

### UI Elements ###
class Button(Rect):
  group = []
  def __init__(self, top,left,w=100,h=60,bg=DARK_BTN, outline_width=1,outline_color=DARK_BTN_OUTLINE,parent=None, id=None):
    super().__init__(top,left,w,h,bg,outline_width,outline_color,parent)
    self.group.append(self)
    self.id = id
    self.top = top
    self.left = left
    self.w = w
    self.h = h
    self.bg = bg
    self.bg_hover = bg.lerp(WHITE, 0.3)
    self.surf = pygame.Surface((w,h))
    self.surf.fill(self.bg)
    self.hovered = False
    self.selected = False


  def draw(self, surf):
    self.surf.fill(self.bg)
    self.outline()
    pygame.draw.circle(surf, WHITE, (self.left,self.top), 1)
    surf.blit(self.surf, (self.left, self.top))


  def update(self, dt):
    self.bg = self._bg

    self.collide(self.parent.game.pointer.xy)
    if self.hovered:
      self.bg = self.bg_hover
      self.parent.game.pointer.enter_button(self)

  def collide(self, point):
    x, y = point - (self.parent.left, self.parent.top)
    self.hovered = x > self.left and x < self.right and y > self.top and y < self.bot
    if self.hovered:
      self.parent.game.debug_panel.print(f'top-left xy: {(self.left,self.top)}, parent xy: {(self.parent.left,self.parent.top)}')
    return self.hovered

  def click(self):
    print('button', self.id)
    self.selected = True

  def on_click(self):
    pass

class ControlPanelButton(Button):
  group = []
  punched = None
  def update(self,dt):
    super().update(dt)
    if ControlPanelButton.punched == self.id:
      self.outline_width = 3
      self.outline_color = DARK_BG.lerp(BLACK,.5)
    else:
      self.outline_width = self._outline_width
      self.outline_color = self._outline_color

  def click(self):
    print('control panel button', self.id)
    ControlPanelButton.punched = self.id
    self.parent.game.recorder.set_active_signal(self.id) # testing

class ListenerPanelButton(Button):
  group = []
  port_color = DARK_BG.lerp(BLACK, .3)
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.n_ports = 4
    self.ports = [None]*self.n_ports
    self.port_width = 5 # get from cable class (todo)
    self.spacing = get_spacing(self.h, s=self.port_width, n=len(self.ports))
    self.port_locations = list(zip([self.absleft-1]*self.n_ports,
                                    self.abstop + self.port_width/2 + np.array(self.spacing)))
    print(self.abstop)
    print('parent:', self.parent.abstop, self.parent.top)
    print(self.port_locations)

  def first_availible_port(self, signal_bank_id):
    for i, port in enumerate(self.ports):
      if port is None:
        self.ports[i] = signal_bank_id
        return self.port_locations[i]
    return None

  def draw(self, surf):
    # surf is parent window (ListenerPanelWindow)
    self.surf.fill(self.bg)
    self.outline()
    for port_y_offset in self.spacing:
      pos = (0, port_y_offset+2)
      pygame.draw.circle(self.surf, self.port_color.lerp(WHITE,.05), pos, 4)
      pygame.draw.circle(self.surf, self.port_color, pos, 3)
    surf.blit(self.surf, (self.left, self.top))

### end Button classes ###

class Cable:
  COLORS = [pygame.Color('#'+hexstr) for hexstr in ('de4d86','118ab2','ffc857','25a18e','7b2cbf')]
  n_path_samples = 20
  width = 4
  group = deque()#[]
  dropped = []
  MAX_LEN = 1000
  MIN_LEN = 50
  COLOR_INDEX = 0
  MAX_COUNT = 16#5
  active_cable = None
  def __init__(self, game, source=None, dest=None, color=None):
    Cable.group.append(self)
    if len(Cable.group) > Cable.MAX_COUNT:
      kill = Cable.group.popleft()
      del kill
    self.game = game
    self.source = source
    self.dest = dest
    self.dragging = False
    self.plugged = False
    self.startpos = np.array((0,0))
    self.endpos = None
    self.color = color or Cable.choose_color()
    self.path = []
    self.samps = []
    self.x = np.linspace(0,1,Cable.n_path_samples)
    self.poly = np.polynomial.Polynomial([0,0,3,-2])
    self.deriv = np.polynomial.Polynomial([0,6,-6,0])
    angles = np.arctan(self.deriv(self.x))
    self.normals = np.c_[-np.sin(angles), np.cos(angles)]
    self._normals = self.normals.copy()
    self.y = self.poly(self.x)

    self.start() # start at creation time
    #self.y = 3*np.power(self.x, 2) - 2*np.power(self.x, 3)

  @classmethod
  def choose_color(cls):
    color = cls.COLORS[cls.COLOR_INDEX % len(cls.COLORS)]
    cls.COLOR_INDEX += 1
    return color 

  def update_path(self):
      u,v = self.endpos - self.startpos
      if np.linalg.norm((u,v)) > Cable.MAX_LEN:
        self.drop()
        print('dropped cable')
      self.samps = self.startpos + np.c_[u*self.x, v*self.y]# pre-calculate positions (or are unvectorized multiplies faster due to less memory alloc?)
      #self.normals = self._normals*(1/u,1/v)

  def update(self, dt):
    if self.dragging:
      self.endpos = self.game.pointer.xy
      self.update_path()


  def normalize(self, v):
    return v/np.linalg.norm(v)

  def draw(self, surf):
    for i in range(1, Cable.n_path_samples):
      pygame.draw.line(surf, self.color, self.samps[i-1], self.samps[i], Cable.width)
      offset = Cable.width//2
      top_color = self.color.lerp(WHITE, 0.5)
      bot_color = self.color.lerp(BLACK, 0.3)
      #r = Cable.width/2
      #normal_offset_a = self.normalize(self.normals[i-1])*r
      #normal_offset_b = self.normalize(self.normals[i])*r
      #pygame.draw.line(surf, top_color, self.samps[i-1] - normal_offset_a, self.samps[i] - normal_offset_b, 3)
      #pygame.draw.line(surf, top_color, self.samps[i-1] + normal_offset_a, self.samps[i] + normal_offset_b, 1)
      pygame.draw.line(surf, top_color, self.samps[i-1] - (0, offset), self.samps[i] - (0, offset), 3)
      pygame.draw.line(surf, bot_color, self.samps[i-1] + (0, offset), self.samps[i] + (0, offset))



  def get_poly(self): # defunct? maybe use to clean up cable after plugged (too expensive to live calculate? prefer domain stretching?)
    """
    form: p(t) = at^3 + bt^2
    solve for a and b assuming (0,0) as relative origin
    """
    x,y = self.endpos - self.startpos
    a = -2*y/x**3
    b = -3*a*x/2
    self.poly = np.polynomial([0,0,b,a])

  def start(self):
    self.dragging = True
    self.startpos = self.game.pointer.xy
    Cable.active_cable = self

    self.selected_signal_bank = None
    for signal_button in ControlPanelButton.group:
      if signal_button.hovered:
        self.selected_signal_bank = signal_button
        self.startpos[0] = self.selected_signal_bank.absleft + self.selected_signal_bank.w


  def plug(self):
    self.endpos = self.game.pointer.xy
    u,v = self.endpos - self.startpos
    if np.linalg.norm((u,v)) < Cable.MIN_LEN:
      return self.drop()

    # check if cursor is over a listener button
    selected_listener_bank = None
    for listener_bank in ListenerPanelButton.group:
      if listener_bank.hovered:
        selected_listener_bank = listener_bank
        self.endpos = selected_listener_bank.first_availible_port(self.selected_signal_bank.id)
        if self.endpos is None: # no availible ports
          return self.drop()
        #self.endpos[0] = listener_bank.absleft
        #self.endpos[1] = port_index
        self.update_path()
        # todo -> add to listener (or is this handled by patch bay without any extra work?)
        self.game.patch_bay.connect(self.selected_signal_bank.id, selected_listener_bank.id)
        break
    if not selected_listener_bank:
      return self.drop()

    self.dragging = False
    self.plugged = True

  def unplug(self):
    pass

  def drop(self):
    Cable.dropped.append(self)
    Cable.active_cable = None
    self.dragging = False
    self.endpos = None
    #self.samps = []


### Level Config ###
class LevelConfig:
  def __init__(self, game):
    self.game = game
    self.listener_pool = { 
              0: Listener(lambda val: setattr(self.game.level_window,'bg', DARK_BG.lerp(WHITE, val))),
              1: Listener(lambda val: self.game.level_window.ball.set_pos(val)),
              2: Listener(lambda val: self.game.level_window.ball.set_radius(val)),
              3: Listener(lambda val: self.game.level_window.ball.set_arm_radius(val)),
            }

class PatchBay:
  def __init__(self, game, size=10):
    self.game = game
    self.graph = np.zeros((size,size),dtype=int)

  def connect(self, source, dest):
    self.graph[source][dest] = 1

  def disconnect(self, source, dest):
    self.graph[source][dest] = 0

  def get_listeners(self, source_id):
    return self.graph[source_id,:].nonzero()[0]

  def get_sources(self, listener_id):
    return self.graph[:,listener_id].nonzero()[0]

### end Level Config ###

### end UI Elements ###
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

    self.over_button = False
    self.hovered_object = None

  def in_hover_zone(self):
    x, y = self.xy
    if y < self.hover_zone_top or y > self.hover_zone_bot or x < self.hover_zone_left or x > self.hover_zone_right:
      self.hovering = False
      return False
    T = x - self.hover_zone_left
    self.hover_val = rescale(T, mn=0, mx=self.game.hover_zone.w, a=0, b=1)
    self.hovering = True
    return True

  def enter_button(self, button):
    self.over_button = True
    self.hovered_object = button

  def exit_button(self):
    self.over_button = False
    self.hovered_object = None

  def report(self):
    self.xy = np.array(pygame.mouse.get_pos())
    self.game.debug_panel.print(self.xy)
    if self.xy[0] > self.game.W/2:
      pygame.mouse.set_cursor(Pointer.hand)
    elif self.in_hover_zone():
      pygame.mouse.set_cursor(Pointer.hover)
    elif self.over_button:
      pygame.mouse.set_cursor(Pointer.hand)
    else:
      pygame.mouse.set_cursor(Pointer.normal)
    self.exit_button()

### utility funcs ###

def clamp(t, mn=0, mx=1):
  return max(mn, min(mx, t))

def rescale(t, mn, mx, a=0, b=1):
  return a + (b-a)*(t-mn)/(mx-mn)

def lerp(t, a, b):
  return a + t*(b-a)

def get_spacing(total, s, n):
  gaps = n+1
  filled = n*s
  empty = max(0,total - filled)
  gap_size = empty / gaps 
  return [gap_size + (gap_size + s)*i for i in range(n)]

### end utility funcs ###

### game objects ###
class Ball:
  def __init__(self, window, fulcrum, arm_radius, ball_radius=4, phase=0.5, color=WHITE, group=None):
    if group is not None:
      group.append(self)
    self.window = window
    self.fulcrum = np.array(fulcrum)
    self.arm = arm_radius
    self._arm = arm_radius
    self.r = ball_radius
    self._r = ball_radius
    self.set_pos(phase)
    self.color = color

  def set_pos(self, t): # t = [0,1] => [pi,0]
    self.phi = lerp(t, math.pi, 0) #t*math.pi - math.pi/2
    #self.pos = self.fulcrum + self.arm*np.array((math.cos(self.phi), -math.sin(self.phi))) # negative y-component to flip y-axis

  def set_radius(self, t): # t = [0,1] => [4,12]
    self.r = lerp(t, self._r, self._r*3)

  def set_arm_radius(self, t): # t = [0,1] => [4,12]
    self.arm = lerp(t, self._arm, self.window.w/2)
    #self.pos = self.fulcrum + self.arm*np.array((math.cos(self.phi), -math.sin(self.phi))) # move this to update method below

  def update(self, dt):
    self.pos = self.fulcrum + self.arm*np.array((math.cos(self.phi), -math.sin(self.phi))) 

  def draw(self, surf):
    pygame.draw.circle(surf, self.color.lerp(self.window.bg, .4), self.pos, self.r)
    pygame.draw.circle(surf, self.color.lerp(self.window.bg, .8), self.pos, self.r,1)

### end game objects ###

class Game:

  def __init__(self):
    pygame.init()

    self.W = 640*2
    self.H = 480
    self.outline_width = 3

    self.FPS = 60.0
    self.clock = pygame.time.Clock()
    self.frame = 0

    self.screen = pygame.display.set_mode((self.W, self.H))
    self.recorder = Recorder(self)
    self.patch_bay = PatchBay(self)

    # Level Config
    self.level_config = LevelConfig(self)

    # Main Control (Left Screen) Window
    MAIN_BG = DARK_BG.lerp(BLACK,0.1)
    left_window = Window(self, top = 0, left=0, w=self.W//2, h=self.H, bg=MAIN_BG, outline_width=self.outline_width, outline_color=DARK_BG_OUTLINE)

    # Signal Input Window
    hover_w, hover_h = left_window.w *0.8, left_window.h*0.2
    self.hover_zone = HoverWindow(self, top = left_window.h//2 - hover_h//2, left = (left_window.w - hover_w)//2, w=hover_w, h=hover_h,
                                  bg=MID.lerp(AMBER,.3), outline_width=3, outline_color=DARK_BG_OUTLINE.lerp(AMBER,.3) ) 
    left_window.add_child(self.hover_zone)

    # Graph Window
    graph_w, graph_h = left_window.w *0.8, (left_window.h - self.hover_zone.bot - PAD*2)
    self.graph_zone = GraphWindow(self, top = self.hover_zone.bot + PAD, left = self.hover_zone.left, w=graph_w, h=graph_h,
                                  bg=MID, outline_width=1, outline_color=DARK_BG_OUTLINE ) 
    left_window.add_child(self.graph_zone)

    # Control Panel Window
    control_panel_w = self.hover_zone.left - 2*PAD
    control_panel_h = left_window.h - 2*PAD
    self.control_panel = ControlPanelWindow(self, top = PAD, left = PAD, w=control_panel_w, h=control_panel_h,
                                  bg=MID, outline_width=1, outline_color=DARK_BG_OUTLINE ) 
    left_window.add_child(self.control_panel)

    # Listener Panel Window
    listener_panel_w = left_window.w - self.hover_zone.right - 2*PAD
    listener_panel_h = left_window.h - 2*PAD
    self.listener_panel = ListenerPanelWindow(self, top = PAD, left = self.hover_zone.right + PAD, w=listener_panel_w, h=listener_panel_h,
                                  bg=MID, outline_width=1, outline_color=DARK_BG_OUTLINE,parent=left_window )  # TESTing parent
    left_window.add_child(self.listener_panel)

    # Debug Window
    self.debug_panel = DebugWindow(self, top = PAD, left = self.hover_zone.left, w=graph_w, h=graph_h,
                                  bg=MID.lerp(WHITE,0.3), outline_width=1, outline_color=DARK_BG_OUTLINE ) 
    left_window.add_child(self.debug_panel)

    # Level Window (Right Screen)
    self.level_window = LevelWindow(self, top = 0, left=self.W//2, w=self.W//2, h=self.H, bg=DARK_BG, outline_width=self.outline_width, outline_color=DARK_BG_OUTLINE)
    self.windows = [left_window, self.level_window]


    # Pointer
    self.pointer = Pointer(self)


  def draw(self):
    self.screen.fill(BLACK)

    for win in self.windows:
      win.draw(self.screen)

    for cable in Cable.group:
      cable.draw(self.screen)

    #pygame.draw.line(self.screen, RED, (590, 74), (590,84), 1)
    #pygame.draw.line(self.screen, RED, (600, 74), (600,84), 1)
    pygame.display.flip()

  def update(self):
    self.frame += 1
    dt = self.dt
    for event in pygame.event.get():
      if event.type == QUIT:
        self.quit()

      elif event.type == pygame.MOUSEBUTTONDOWN:
        if self.pointer.over_button and isinstance(self.pointer.hovered_object, ControlPanelButton):
          Cable(self)
      elif event.type == pygame.MOUSEBUTTONUP:
        if Cable.active_cable:
           Cable.active_cable.plug()

      elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_q:
            self.quit()
          if event.key == pygame.K_r:
            game.recorder.start_recording()
          elif event.key == pygame.K_p:
            game.recorder.start_playback()
          elif event.key == pygame.K_1:
            ControlPanelButton.group[0].click()
          elif event.key == pygame.K_2:
            ControlPanelButton.group[1].click()
          elif event.key == pygame.K_3:
            ControlPanelButton.group[2].click()
          elif event.key == pygame.K_4:
            ControlPanelButton.group[3].click()



    self.pointer.report()
    self.recorder.update(dt)

    for button in ControlPanelButton.group:
      button.update(dt)

    for button in ListenerPanelButton.group:
      button.update(dt)

    for cable in Cable.group:
      cable.update(dt)

    for cable in Cable.dropped:
      Cable.group.remove(cable)
    Cable.dropped = []

    self.debug_panel.update(dt)

    self.level_window.update(dt) 

  def run(self):
    self.dt = 1/self.FPS
    while True:
      self.update()
      self.draw()
      self.dt = self.clock.tick(self.FPS)

  def quit(self):
        pygame.quit()
        sys.exit() 

class Listener:
  def __init__(self, set_attr_func, mn=0, mx=1):
    self.set_attr_func = set_attr_func
    self.mn = mn
    self.mx = mx

  def notify(self, val):
    self.set_attr_func(lerp(val, self.mn, self.mx))

class Signal:
  stored = { } # id => signal
  def __init__(self, id=0, samples=None, smoothed=None, poly=None, listeners=None):
    self.id = id
    self.samples = samples or []
    self.smoothed = smoothed or []
    self.poly = poly or None
    self.listeners = listeners or []
    self.complete = False

  def reset(self):
    self.samples = []
    self.smoothed = []
    self.poly = None
    self.complete = False

class Recorder:
  def __init__(self, game):
    self.game = game
    self.sr = 24 # samples per second
    self.dur = 5 # capture seconds
    self.T = 1/self.sr # period in seconds
    self.n = int(self.sr*self.dur)
    self.n_smooth_samples = int(self.game.FPS*self.dur)

    self.recording = False

    self.playing = False
    self.playback_progress = 0
    self.playback_val = 0

    self.active_signal = Signal() # keep an 'active signal' and swap out with others as user clicks through

  @property
  def samples(self):
    return self.active_signal.samples
  @property
  def smoothed(self):
    return self.active_signal.smoothed
  @property
  def poly(self):
    return self.active_signal.poly
  @property
  def listeners(self): # testing..maybe don't use this property in this class
    return self.active_signal.listeners
  
  
  def set_active_signal(self, id):
    #self.end_playback() # TESTING! probably remove so as to not interrupt playback when switching

    if self.recording:
      self.cancel_recording()

    print('set active signal to', id)
    signal = Signal.stored.get(id)
    if not signal: # create new Signal
      signal = Signal(id=id)
      Signal.stored[id] = signal
      self.end_playback() # testing
    self.active_signal = signal #Signal.stored.get(id, Signal(id=id))
  
  def update(self, dt): # dt in ms
    if self.recording:
      t = time.time()
      elapsed = t-self.t0

      #for listener in self.listeners: 
      for listener_id in self.game.patch_bay.get_listeners(self.active_signal.id):
        listener = self.game.level_config.listener_pool[listener_id]
        listener.notify(self.game.pointer.hover_val)

      if elapsed >= self.T*self.i:
        self.sample()

    elif self.playing: 
      t = time.time()
      elapsed = t-self.playback_t0
      self.playback_progress = clamp((elapsed/self.dur),0,1)

      for listener_id, listener in self.game.level_config.listener_pool.items():
        summed_signal = self.sum_signals(listener_id)
        if summed_signal is not None:
          playback_index = int(self.playback_progress*(len(summed_signal) - 1))
          playback_val = summed_signal[playback_index]
          listener.notify(playback_val)

      if elapsed >= self.dur: 
        self.end_playback()

  def register_listener(self, listener):
    self.listeners.append(listener) # todo: attach listeners to banks (ListenerButtons) not signals
                                    # also todo: switch from list to dictionary / add remove_listener method

  def sum_signals(self, listener_id):
    summed_signal = np.zeros(self.n_smooth_samples) 
    n_summed = 0
    source_signal_ids = self.game.patch_bay.get_sources(listener_id)
    
    for signal_id in source_signal_ids: 
      sig = Signal.stored.get(signal_id) 
      if not sig or not sig.complete:
        continue
      summed_signal += sig.smoothed 
      n_summed += 1

    if n_summed == 0:
      return None
    return summed_signal.clip(0,1)

  def sample(self):
    T = self.game.pointer.hover_val
    self.samples.append(T)
    self.i += 1
    if self.i == self.n: # finished sampling 
      self.finish()

  def finish(self):
      self.recording = False
      total_elapsed = time.time() - self.t0
      print(f'finished: took {self.n} samples in {total_elapsed} seconds.') 
      X = np.linspace(0, self.dur, self.n, True)
      Y = self.samples
      DOWNSAMPLE = 2
      X = np.r_[X[:-1][::DOWNSAMPLE],X[-1]]
      Y = np.r_[Y[:-1][::DOWNSAMPLE],Y[-1]]
      self.active_signal.poly = piecewise3poly(X,Y)
      t = np.linspace(0, self.dur, self.n_smooth_samples, False)
      self.active_signal.smoothed = [clamp(self.poly(i),0,1) for i in t]
      self.active_signal.complete = True

  def cancel_recording(self):
    self.active_signal.reset()
    self.recording = False

  def start_recording(self):
    self.recording = True
    self.active_signal.reset()
    self.i = 0
    self.t0 = time.time()
    self.sample()
    print('started recording')

  def start_playback(self):
    #if not self.active_signal.complete: # (note: check handled in update now..unneeded here.)
    #  return
    self.playing = True
    self.playback_t0 = time.time()

  def end_playback(self):
    self.playing = False

game = Game()
game.run()