import FreeCAD, Part, Mesh
import glob, os

input_dir = "geometries/IGS/"
output_dir = "DSYHS_STL/"

os.makedirs(output_dir, exist_ok=True)

for file in glob.glob(input_dir + "*.igs"):
    shape = Part.read(file)

    doc = FreeCAD.newDocument()
    obj = doc.addObject("Part::Feature","hull")
    obj.Shape = shape

    mesh = Mesh.Mesh()
    mesh.addFacets(obj.Shape.tessellate(0.1))

    out_name = os.path.basename(file).replace(".igs",".stl")
    mesh.write(output_dir + out_name)

print("Done: STL pack created")