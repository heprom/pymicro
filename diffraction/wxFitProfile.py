#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, wx, pickle
from view.wxPlotPanel import PlotPanel
from wxRadialProfile import ProfilePanel
from fit import *
from wx.lib.masked import NumCtrl
from wx.lib.masked.numctrl import EVT_NUM
import wx.lib.newevent
import numpy, waxd
    
class FitProfilePanel(ProfilePanel):
  '''The FitProfilePanel extends ProfilePanel to display both profile 
  and fit functions in a matplotlib canvas.'''

  def __init__(self, parent, **kwargs):
    # initialize Panel
    self.parent = parent
    self.funcs = []
    self.plot_fit = False
    # initiate plotter
    ProfilePanel.__init__(self, parent, color=None, dpi=None, **kwargs)

  def update_funcs(self, funcs):
    self.funcs = funcs
    self.draw()
    
  def setPlotFit(self, bool):
    self.plot_fit = bool
    
  def draw(self):
    '''Draw profile data.'''
    if not hasattr(self, 'subplot'):
      self.subplot = self.figure.add_subplot(111)
    self.subplot.clear()
    if self.profile == None:
      return
    self.subplot.plot(self.profile[0], self.profile[1])
    x = self.profile[0]
    fit = numpy.zeros_like(x)
    for func in self.funcs:
      y = func.compute(x)
      fit += y
      if func.plot:
        self.subplot.plot(x, y)
    # plot complete fit if needed
    if self.plot_fit:
      print 'plotting fit'
      self.subplot.plot(x,fit)
    self.Layout()

class editFitFuncPanel(wx.Panel):

  def __init__( self, parent, func=None, **kwargs ):
    # initialize Panel
    if 'id' not in kwargs.keys():
      kwargs['id'] = wx.ID_ANY
    if 'style' not in kwargs.keys():
      kwargs['style'] = wx.NO_FULL_REPAINT_ON_RESIZE
    wx.Panel.__init__( self, parent, **kwargs )
    self.paramsPanel = wx.Panel(self, id=wx.ID_ANY)
    self.func = func
    self.auto_update = False
    sizer = wx.BoxSizer(wx.VERTICAL)
    #self.paramsPanel.SetBackgroundColour('Green')
    grid = wx.GridSizer(3+len(func.params), 3, 3, 3)
    self.fname = wx.TextCtrl(self.paramsPanel, -1, func.name)
    self.typeCombo = wx.ComboBox(self.paramsPanel, id=421, choices=fitFunc.types, style=wx.CB_DROPDOWN)
    self.Bind(wx.EVT_COMBOBOX, self.OnTypeSelected, id=421)
    # unified gui for all fitFunc types
    self.typeCombo.Select(self.func.type)
    grid.AddMany([(wx.StaticText(self.paramsPanel, wx.ID_ANY, 'Type:'), 0, wx.ALIGN_CENTER), 
      (self.typeCombo, 0, wx.ALIGN_CENTER | wx.EXPAND),
      (wx.StaticText(self.paramsPanel, wx.ID_ANY, ''), 0, wx.ALIGN_CENTER),
      (wx.StaticText(self.paramsPanel, wx.ID_ANY, 'Name:'), 0, wx.ALIGN_CENTER),
      (self.fname, 0, wx.TE_CENTER | wx.EXPAND),
      (wx.StaticText(self.paramsPanel, wx.ID_ANY, ''), 0, wx.ALIGN_CENTER),
      ])
    # Add all NumCtrl fields
    self.numCtrls = []
    for p in range(len(func.params)):
      self.numCtrls.append(NumCtrl(self.paramsPanel, -1, fractionWidth = 2, \
        validator = wx.DefaultValidator, signedForegroundColour = "Black", \
        emptyBackgroundColour = "Red", selectOnEntry = True))
      #disabled for now self.Bind(EVT_NUM, self.OnParamChanged, self.numCtrls[p])
      grid.AddMany([
        (wx.StaticText(self.paramsPanel, wx.ID_ANY, fitFunc.par_labels[func.type][p]), 0, wx.ALIGN_CENTER),
        (self.numCtrls[p], 1, wx.TE_CENTER | wx.EXPAND | wx.ALIGN_CENTER),
        (wx.StaticText(self.paramsPanel, wx.ID_ANY, fitFunc.par_units[func.type][p]), 0, wx.ALIGN_CENTER),
        ])
    self.paramsPanel.SetSizer(grid)
    sizer.Add(self.paramsPanel, 1, wx.EXPAND)
    sizer.AddSpacer(5)
    sizer.Add(wx.StaticLine(self, -1, (25, 50), (300,1)), 0, wx.EXPAND)
    sizer.AddSpacer(5)
    buttonBox = wx.BoxSizer(wx.HORIZONTAL)
    self.apply = wx.Button(self, 110, '&Apply')
    self.reset = wx.Button(self, 120, '&Reset changes')
    buttonBox.Add(wx.CheckBox(self, -1, 'Auto &update'), proportion=1, flag=wx.EXPAND, border=3)
    buttonBox.Add(self.apply, proportion=1, flag=wx.EXPAND, border=3)
    buttonBox.Add(self.reset, proportion=1, flag=wx.EXPAND, border=3)
    self.Bind(wx.EVT_CHECKBOX, self.OnAutoUpdate, id=-1)
    self.Bind(wx.EVT_BUTTON, self.OnApplyChanges, id=110)
    self.Bind(wx.EVT_BUTTON, self.OnResetChanges, id=120)
    # now display param values in TextCrtls
    self.OnResetChanges(None)
    sizer.Add(buttonBox)
    self.SetSizer(sizer)

  def DisplayParams(self):
    self.fname.SetValue(self.func.name)
    for p in range(len(self.func.params)):
      self.numCtrls[p].SetValue(self.func.params[p])

  def OnTypeSelected(self, event):
    if self.typeCombo.GetSelection() == self.func.type:
      return
    # erase the current function with an empty one of the appropriate type
    evt = wx.PyCommandEvent(wx.EVT_BUTTON.typeId, self.GetId())
    evt.data = fitFunc(type=self.typeCombo.GetSelection())
    wx.PostEvent(self.GetEventHandler(), evt)
    
  def OnParamChanged(self, event):
    print 'param changed'
    #TODO fire a EVT_PLOT or something here and EVT_NAME otherwise
    if self.auto_update:
      self.OnApplyChanges(event)
    
  def OnAutoUpdate(self, event):
    self.auto_update = not self.auto_update
    self.apply.Enable(not self.auto_update)
    self.reset.Enable(not self.auto_update)
    
  '''Update function with name and values in NumCtrl boxes.'''
  def OnApplyChanges(self, event):
    self.func.name = self.fname.GetValue()
    for p in range(len(self.func.params)):
      self.func.params[p] = self.numCtrls[p].GetValue()
    # fire a new event to notify function change
    evt = wx.PyCommandEvent(wx.EVT_BUTTON.typeId, self.GetId())
    evt.data = self.func
    wx.PostEvent(self.GetEventHandler(), evt)

  '''Reset all form fields with actual values of function parameters.'''
  def OnResetChanges(self, event):
    self.fname.SetValue(self.func.name)
    for p in range(len(self.func.params)):
      self.numCtrls[p].SetValue(self.func.params[p])
      #self.numCtrls[p].SetValue(float(self.func.params[p]))

class wxFitFrame(wx.Frame):

  def __init__(self, file=None):
    wx.Frame.__init__(self, None, wx.ID_ANY, title = u"wx Fit Profile")
    print 'in init wxFitFrame file=',file
    self.SetSize((840, 680))
    self.statusbar = self.CreateStatusBar()
    self.panel = wx.Panel(self, -1)
    self.sizer = wx.BoxSizer(wx.VERTICAL)

    # fit function list
    self.fitList = []
    #self.fitList.append(fitFunc(name='incoherent scattering', type='linear', params=[-40., 170000.]))
    #self.fitList.append(fitFunc(name='Zno peak 1', type='pearson7', params=[400000., 2790., 25., 2.]))
    self.buildFitFuncNameList()
    
    # * child 1: plot panel
    self.profilePanel = FitProfilePanel(self.panel, id=wx.ID_ANY)
    self.profilePanel.update_funcs(self.fitList)
    self.sizer.Add(self.profilePanel, proportion=2, flag=wx.EXPAND | wx.ALL, border=5)
    # * child 2: fit panel
    self.fitPanel = wx.Panel(self.panel, wx.ID_ANY, style=wx.SUNKEN_BORDER)
    # ** list panel
    listPanel = wx.Panel(self.fitPanel, wx.ID_ANY, style=wx.SUNKEN_BORDER)
    self.list = wx.CheckListBox(listPanel, id=wx.ID_ANY, style=wx.LB_SINGLE, choices=self.fitFuncNameList)
    listPanelBox = wx.BoxSizer(wx.VERTICAL)
    listPanel.SetSizer(listPanelBox)
    listPanelBox.Add(self.list, 1, wx.EXPAND | wx.ALL, 5)
    self.Bind(wx.EVT_LISTBOX, self.OnSelectListItem, self.list)
    self.Bind(wx.EVT_CHECKLISTBOX, self.OnCheckListItem, self.list)
    # ** button panel below list
    #listButtonPanel = wx.Panel(self.listPanel, wx.ID_ANY, style=wx.SUNKEN_BORDER)
    listButtonBox = wx.BoxSizer(wx.HORIZONTAL)
    listButtonBox.Add(wx.Button(listPanel, 41, "Add &func"), proportion=1, flag=wx.EXPAND)
    listButtonBox.Add(wx.Button(listPanel, 42, '&Delete'), proportion=1, flag=wx.EXPAND)
    self.plot_fit = wx.CheckBox(listPanel, 43, '&Plot fit')
    listButtonBox.Add(self.plot_fit, proportion=1, flag=wx.EXPAND)
    self.Bind(wx.EVT_BUTTON, self.OnAddFunc, id=41)
    self.Bind(wx.EVT_BUTTON, self.OnDelFunc, id=42)
    self.Bind(wx.EVT_CHECKBOX, self.OnCheckPlotFit, id=43)
    listPanelBox.Add(listButtonBox, proportion=0, flag=wx.EXPAND)
    #self.SetSizer(sizer)

    # ** edit panel
    self.editPanel = wx.Panel(self.fitPanel, wx.ID_ANY)
    self.editBox = wx.StaticBox(self.editPanel, -1, 'Edit')
    editSizer = wx.StaticBoxSizer(self.editBox, wx.VERTICAL)
    self.editPanel.SetSizer(editSizer)
    # ** button panel
    buttonBox = wx.BoxSizer(wx.VERTICAL)
    buttonBox.Add(wx.Button(self.fitPanel, 10, '&Load profile'), proportion=0, flag=wx.EXPAND, border=3)
    buttonBox.Add(wx.Button(self.fitPanel, 11, '&Load fit'), proportion=0, flag=wx.EXPAND, border=3)
    buttonBox.Add(wx.Button(self.fitPanel, 12, '&Save fit'), proportion=0, flag=wx.EXPAND, border=3)
    buttonBox.Add(wx.Button(self.fitPanel, 13, '&Quit'), proportion=0, flag=wx.EXPAND, border=3)
    self.Bind(wx.EVT_BUTTON, self.OnLoadProfile, id=10)
    self.Bind(wx.EVT_BUTTON, self.OnLoadFit, id=11)
    self.Bind(wx.EVT_BUTTON, self.OnSaveFit, id=12)
    self.Bind(wx.EVT_BUTTON, self.OnQuit, id=13)
    # * layout child 2
    self.fitBox = wx.BoxSizer(wx.HORIZONTAL)
    self.fitBox.Add(buttonBox, 0, wx.EXPAND | wx.ALL, 5)
    self.fitBox.Add(listPanel, 1, wx.EXPAND | wx.ALL, 5)
    self.fitBox.Add(self.editPanel, 2, wx.EXPAND | wx.ALL, 5)
    self.fitPanel.SetSizer(self.fitBox)
    self.sizer.Add(self.fitPanel, proportion=1.5, flag=wx.EXPAND | wx.ALL, border=5)

    # final settings
    self.panel.SetSizer(self.sizer)
    # load profile if a file path was provided
    print 'before calling OnLoadProfile file=',file
    if file != None:
      self.OnLoadProfile(None, file)
    self.Centre()
    
  def buildFitFuncNameList(self):
    self.fitFuncNameList = []
    for fitFunc in self.fitList:
      print fitFunc.name
      self.fitFuncNameList.append(fitFunc.name)

  '''Append new Fitunc to the list.'''
  def OnAddFunc(self, event):
  	self.fitList.append(fitFunc(name='new function', type=fitFunc.linear, params=[0., 0.]))
  	self.buildFitFuncNameList()
  	self.list.Set(self.fitFuncNameList)
  	self.list.Refresh()
    
  '''Delete selected FitFunc from the list.'''
  def OnDelFunc(self, event):
  	self.fitList.pop(self.list.GetSelection())
  	self.profilePanel.update_funcs(self.fitList)
  	self.buildFitFuncNameList()
  	self.list.Set(self.fitFuncNameList)
  	self.list.Refresh()
  
  '''Plot complete fit on profile panel.'''
  def OnCheckPlotFit(self, event):
    self.profilePanel.setPlotFit(self.plot_fit.IsChecked())
    self.profilePanel.draw()
    
  '''One item in the list was checked to be plotted or to be removed 
  from the plot panel. The boolean plot attribute of each function in 
  the list is updated.'''
  def OnCheckListItem(self, event):
  	#print '\n item checked: ', self.list.GetStringSelection(), 'at index', self.list.GetSelection()
  	for i in range(len(self.fitList)):
  	  #do not work# self.fitList[self.list.GetSelection()].plot = self.list.IsChecked(self.list.GetSelection())
  	  self.fitList[i].plot = self.list.IsChecked(i)
  	self.profilePanel.update_funcs(self.fitList)
  	
  def OnSelectListItem(self, event):
  	#print 'item selected: ', self.list.GetStringSelection(), 'at index', self.list.GetSelection()
  	self.editPanel.Destroy()
  	self.editPanel = editFitFuncPanel(self.fitPanel, func=self.fitList[self.list.GetSelection()], id=wx.ID_ANY, style=wx.SUNKEN_BORDER)
  	self.Bind(wx.EVT_BUTTON, self.OnItemChanged, self.editPanel)
  	self.fitBox.Add(self.editPanel, 2, wx.EXPAND | wx.ALL, 5)
  	self.fitBox.Layout()
  	
  def OnItemChanged(self, event):
    print 'ITEM CHANGED!!'
    selected = self.list.GetSelection()
    print 'selected:',selected
    checked = self.list.IsChecked(selected)
    # look if fitfunc type just changed
    changed = self.fitList[self.list.GetSelection()].type != event.data.type
    self.fitList[self.list.GetSelection()] = event.data
    self.profilePanel.update_funcs(self.fitList)
    self.list.SetString(selected, event.data.name)
    if changed:
      self.OnSelectListItem(event)
    if checked:
      self.list.Check(selected)
      
  def OnLoadProfile(self, event, file=''):
    if file == '':
      # bring up a file chosser dialog
      wildcard = "Profile files (*.profile)|*.profile|" "All files (*.*)|*.*"
      dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.OPEN)
      if dialog.ShowModal() == wx.ID_OK:
        print dialog.GetPath()
        file = dialog.GetPath()
      dialog.Destroy()
    if file != '':
      print 'in OnLoadProfile file=',file
      self.file = file
      print 'in OnLoadProfile self.file=',self.file
      # right now in save_profile, we save a two column file
      self.profile = pickle.load(open(file, 'r'))
      self.profilePanel.update_profile(self.profile)
      self.statusbar.SetStatusText('Profile loaded from %s ' % file)
      # load ident parameters if .ident file exists
      if os.path.exists(file + '.ident'):
        self.OnLoadFit(event, file=file + '.ident')
  
  '''Load fitting parameters.'''
  def OnLoadFit(self, event, file=''):
    if file == '':
      # bring up a file chosser dialog
      wildcard = "Identification files (*.ident)|*.ident|" "All files (*.*)|*.*"
      dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.OPEN)
      if dialog.ShowModal() == wx.ID_OK:
        print dialog.GetPath()
        file = dialog.GetPath()
      dialog.Destroy()
    if file != '':
      print 'loading fitting parameters from ', file
      self.fitList = pickle.load(open(file, 'r'))
      # hack around for compatibility
      for f in self.fitList:
        if f.type == 'linear': f.type = fitFunc.linear
        elif f.type == 'pearson7': f.type = fitFunc.pearson7
      self.profilePanel.update_funcs(self.fitList)
      self.buildFitFuncNameList()
      self.list.Set(self.fitFuncNameList)
      # check list according to fitList
      for i, f in enumerate(self.fitList):
        self.list.Check(i, f.plot)
      self.list.Refresh()
    
  '''Save fitting parameters to a file.'''
  def OnSaveFit(self, event):
    fname = self.file + '.ident'
    print 'saving fit parameters to ', fname
    pickle.dump(self.fitList, open(fname, 'w'), 1)

  def OnQuit(self, event):
    self.Close()

class wxFitProfile(wx.App):

  def __init__(self, file):
    self.file = file
    print 'in init wxFitProfile, self.file=',self.file
    wx.App.__init__(self)

  def OnInit(self):
    '''Init wxFitProfile application.'''
    print 'in OnInit wxFitProfile, self.file=',self.file
    frame = wxFitFrame(file)
    frame.Show()
    # set the main window
    self.SetTopWindow(frame)
    return True

if __name__ == '__main__':
  data_dir = '/home/proudhon/data/20080545/nr_04/'
  name = 'nr_04_022.image'
  name = 'nr_04_092.image'
  name = 'nr_04_094.image'
  name = 'nr_04_096.image'
  name = 'nr_04_098.image'
  name = 'nr_04_092.image_am'
  file = data_dir + name + '.profile'
  name = '/home/20090552/2010_0205/PVA_1.edf'
  file = name + '.profile'

  # create application instance
  app = wxFitProfile(file)
  # start main event loop
  app.MainLoop()
