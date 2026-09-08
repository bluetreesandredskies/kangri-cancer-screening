# \# Dataset Notes

# 

# This project uses five classes: \*\*squamous cell carcinoma\*\* (including its

# invasive/in-situ/NOS variants), \*\*actinic keratosis\*\*, \*\*nevus\*\*,

# \*\*seborrheic keratosis\*\*, and \*\*healthy\*\*. The first four come from ISIC and

# represent a mix of high-risk lesion types (squamous cell carcinoma and

# actinic keratosis are malignant/pre-malignant) and common benign look-alikes

# (nevus, seborrheic keratosis) the model needs to rule out.

# 

# These four lesion classes were chosen as the closest public analogue to

# kangri-cancer's precursor pattern, chronic erythema ab igne progressing

# toward squamous cell carcinoma on thermally-exposed skin. No public dataset

# labels erythema ab igne or kangri-specific lesions directly, so this is the

# nearest proxy available in ISIC's public archive.

# 

# The \*\*healthy\*\* class — smartphone photos of skin with no visible lesion —

# does not come from ISIC, since ISIC is a lesion-image archive and has no

# "no lesion" category at all. Instead it's sourced from the \*\*MCSI dataset\*\*

# (Campana et al.), which was built for exactly this kind of everyday,

# non-clinical skin photography. Without a healthy class, the model would

# never see a true negative and couldn't distinguish "no lesion" from "lesion

# present," which matters a lot for a screening tool meant to run on ordinary

# phone photos rather than dermoscopic images.

# 

# \*\*Attribution:\*\* Contains images from the MCSI dataset (Campana, M.G.,

# Colussi, M., Delmastro, F., Mascetti, S., Pagani, E. —

# https://doi.org/10.5281/zenodo.8360076, CC-BY 4.0).

# 

# \*\*Honesty note:\*\* this pipeline trains a general skin-lesion risk classifier

# on public ISIC and MCSI data. It is a proof-of-concept pilot demonstrating

# that this kind of image classification approach is viable, not a

# kangri-cancer-specific detector, and it has not been validated against any

# kangri-cancer or erythema-ab-igne cases. Treat any output from this model as

# a research demo, not a diagnostic tool.

