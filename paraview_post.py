#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ParaView post-processing of a STAPpy result (.vtk).

Loads the unstructured grid, deforms it by the nodal Displacement vector
(Warp By Vector), colours it by a chosen field and saves a screenshot.

Run with ParaView's own interpreter (it ships paraview.simple):
    pvpython paraview_post.py  data/Bridge-1.vtk  bridge.png  --scale 20 --field Displacement_Magnitude
    pvbatch  paraview_post.py  data/Bridge-1.vtk  bridge.png            # offscreen / headless
Or paste the body into the ParaView GUI  View > Python Shell.

API reference: https://www.paraview.org/paraview-docs/nightly/python/
"""
import argparse
from paraview.simple import (
    OpenDataFile, WarpByVector, GetActiveViewOrCreate, Show, Hide, ColorBy,
    GetColorTransferFunction, GetScalarBar, ResetCamera, Render, SaveScreenshot,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk")
    ap.add_argument("png", nargs="?", default="result.png")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="displacement magnification (0 = auto)")
    ap.add_argument("--field", default="Displacement_Magnitude",
                    help="Displacement_Magnitude | Stress_Measure | Displacement")
    ap.add_argument("--res", default="1600,900")
    args = ap.parse_args()

    reader = OpenDataFile(args.vtk)

    # auto magnification: ~5% of the model size / max displacement
    scale = args.scale
    if scale <= 0.0:
        b = reader.GetDataInformation().GetBounds()
        diag = ((b[1] - b[0])**2 + (b[3] - b[2])**2 + (b[5] - b[4])**2) ** 0.5
        di = reader.GetPointDataInformation().GetArray("Displacement_Magnitude")
        umax = di.GetComponentRange(0)[1] if di else 0.0
        scale = (0.05 * diag / umax) if umax > 0 else 1.0

    warp = WarpByVector(Input=reader)
    warp.Vectors = ["POINTS", "Displacement"]
    warp.ScaleFactor = scale

    view = GetActiveViewOrCreate("RenderView")
    Hide(reader, view)
    display = Show(warp, view)
    display.SetRepresentationType("Surface With Edges")

    loc = "CELLS" if args.field == "Stress_Measure" else "POINTS"
    ColorBy(display, (loc, args.field))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(view, True)
    GetScalarBar(GetColorTransferFunction(args.field), view).Title = args.field

    view.ViewSize = [int(x) for x in args.res.split(",")]
    ResetCamera(view)
    Render()
    SaveScreenshot(args.png, view, ImageResolution=view.ViewSize)
    print("saved", args.png, " (warp scale = %.4g, colour = %s)" % (scale, args.field))


if __name__ == "__main__":
    main()
