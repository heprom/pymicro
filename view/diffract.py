#!/usr/bin/env python

# This is a simple volume rendering example that uses a
# vtkVolumeRayCast mapper

import vtk

# Create the standard renderer, render window and interactor
ren = vtk.vtkRenderer()
ren.SetBackground(0.329412, 0.34902, 0.427451) #Paraview blue
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Create the reader for the data
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName("/home/proudhon/mayavi.vtk")

# Create transfer mapping scalar value to opacity
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0, 0.0)
opacityTransferFunction.AddPoint(200, 0.1)
opacityTransferFunction.AddPoint(255, 1.0)

# Create transfer mapping scalar value to color
colorTransferFunction = vtk.vtkColorTransferFunction()
#colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
#colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
#colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
#colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
#colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)
colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 1.0)
colorTransferFunction.AddRGBPoint(255.0, 1.0, 0.0, 0.0)

# The property describes how the data will look
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()

# The mapper / ray cast function know how to render the data
compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
volumeMapper = vtk.vtkVolumeRayCastMapper()
volumeMapper.SetVolumeRayCastFunction(compositeFunction)
volumeMapper.SetInputConnection(reader.GetOutputPort())

# The volume holds the mapper and the property and
# can be used to position/orient the volume
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren.AddVolume(volume)
ren.SetBackground(1, 1, 1)
renWin.SetSize(600, 600)
renWin.Render()

def CheckAbort(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)
 
renWin.AddObserver("AbortCheckEvent", CheckAbort)

iren.Initialize()
renWin.Render()
iren.Start()
