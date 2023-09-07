import time
import sys
import math
 
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

  @property
  def bot(self):
    return self.top + self.h - 1
  
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

class LevelWindow(Window):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sprites = []
    ball_color = RED.lerp(DARK_BG, .5)
    self.ball = Ball(self, fulcrum=(self.w//2, self.h), arm_radius=self.w//3, ball_radius=12, phase=0.5, color=ball_color, group=self.sprites)
    self.game.recorder.register_listener(Listener(lambda val: self.ball.set_pos(val)))

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

  @property
  def bot(self):
    return self.top + self.h - 1
  @property
  def right(self):
    return self.left + self.w - 1

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
      self.parent.game.pointer.enter_button()

  def collide(self, point):
    x, y = point - (self.parent.left, self.parent.top)
    self.hovered = x > self.left and x < self.right and y > self.top and y < self.bot
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

  def in_hover_zone(self):
    x, y = self.xy
    if y < self.hover_zone_top or y > self.hover_zone_bot or x < self.hover_zone_left or x > self.hover_zone_right:
      self.hovering = False
      return False
    T = x - self.hover_zone_left
    self.hover_val = rescale(T, mn=0, mx=self.game.hover_zone.w, a=0, b=1)
    self.hovering = True
    return True

  def enter_button(self):
    self.over_button = True

  def exit_button(self):
    self.over_button = False

  def report(self):
    self.xy = np.array(pygame.mouse.get_pos())
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

### end utility funcs ###

### game objects ###
class Ball:
  def __init__(self, window, fulcrum, arm_radius, ball_radius=4, phase=0.5, color=WHITE, group=None):
    if group is not None:
      group.append(self)
    self.window = window
    self.fulcrum = np.array(fulcrum)
    self.arm = arm_radius
    self.r = ball_radius
    self.set_pos(phase)
    self.color = color

  def set_pos(self, t): # t = [0,1] => [pi,0]
    self.phi = lerp(t, math.pi, 0) #t*math.pi - math.pi/2
    self.pos = self.fulcrum + self.arm*np.array((math.cos(self.phi), -math.sin(self.phi))) # negative y-component to flip y-axis

  def update(self, dt):
    pass

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


    self.screen = pygame.display.set_mode((self.W, self.H))
    self.recorder = Recorder(self)

    # Main Control (Left Screen) Window
    left_window = Window(self, top = 0, left=0, w=self.W//2, h=self.H, bg=DARK_BG, outline_width=self.outline_width, outline_color=DARK_BG_OUTLINE)

    # Signal Input Window
    hover_w, hover_h = left_window.w *0.8, left_window.h*0.2
    self.hover_zone = HoverWindow(self, top = left_window.h//2 - hover_h//2, left = (left_window.w - hover_w)//2, w=hover_w, h=hover_h,
                                  bg=MID, outline_width=1, outline_color=DARK_BG_OUTLINE ) 
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

    # Level Window (Right Screen)
    right_window = LevelWindow(self, top = 0, left=self.W//2, w=self.W//2, h=self.H, bg=DARK_BG, outline_width=self.outline_width, outline_color=DARK_BG_OUTLINE)
    self.windows = [left_window, right_window]

    # Pointer
    self.pointer = Pointer(self)

    # Recorder
    #self.recorder.register_listener(Listener(lambda val: setattr(right_window,'bg', DARK_BG.lerp(WHITE, val))))

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
          elif event.key == pygame.K_6:
            self.recorder.register_listener(Listener(lambda val: setattr(self.windows[1],'bg', DARK_BG.lerp(WHITE, val))))



    self.pointer.report()
    self.recorder.update(dt)

    for button in ControlPanelButton.group:
      button.update(dt)

  def run(self):
    self.dt = 1/self.FPS
    while True:
      self.update()
      self.draw()
      self.dt = self.clock.tick(self.FPS)

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

  def reset(self):
    self.samples = []
    self.smoothed = []
    self.poly = None

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

    self.active_signal = Signal()
    # TODO: keep an 'active signal' and swap out with others as user clicks through

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
    self.end_playback() # TESTING! probably remove so as to not interrupt playback when switching

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

      for listener in self.listeners: # testing this here...should really be at the per-signal level
        listener.notify(self.game.pointer.hover_val)

      if elapsed >= self.T*self.i:
        self.sample()

    elif self.playing: # TODO: handle all listeners for all Signals.stored signals, not just active_signal and its listeners
      t = time.time()
      elapsed = t-self.playback_t0
      self.playback_progress = clamp((elapsed/self.dur),0,1)

      for sig in Signal.stored.values():
        playback_index = int(self.playback_progress*(len(sig.smoothed) - 1))
        playback_val = sig.smoothed[playback_index]
        for listener in sig.listeners:
          listener.notify(playback_val)

      #playback_index = int(self.playback_progress*(len(self.smoothed) - 1))
      #self.playback_val = self.smoothed[playback_index]
      #for listener in self.listeners:
      #  listener.notify(self.playback_val)

      if elapsed >= self.dur: 
        self.end_playback()

  def register_listener(self, listener):
    self.listeners.append(listener) # todo: switch from list to dictionary / add remove_listener method

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
    if len(self.active_signal.samples) != self.n: # Signal not finished recording (TESTING)
      return 
    self.playing = True
    self.playback_t0 = time.time()

  def end_playback(self):
    self.playing = False

game = Game()
game.run()