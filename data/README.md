# Dataset Notes

This project uses four ISIC diagnosis categories: **squamous cell carcinoma**
(including its invasive/in-situ/NOS variants), **actinic keratosis**, **nevus**,
and **seborrheic keratosis**. Squamous cell carcinoma and actinic keratosis
represent malignant and pre-malignant lesion patterns, while nevus and
seborrheic keratosis are common benign lesions — together they give the model
a mix of high-risk and low-risk classes to learn to separate.

These four were chosen as the closest public analogue to kangri-cancer's
precursor pattern, chronic erythema ab igne progressing toward squamous cell
carcinoma on thermally-exposed skin. No public dataset labels erythema ab
igne or kangri-specific lesions directly, so this is the nearest proxy
available in ISIC's public archive: a mix of the malignant endpoint (SCC),
a common precursor-type lesion (actinic keratosis), and benign look-alikes
the model needs to rule out.

**Honesty note:** this pipeline trains a general skin-lesion risk classifier
on public ISIC data. It is a proof-of-concept pilot demonstrating that this
kind of image classification approach is viable, not a kangri-cancer-specific
detector, and it has not been validated against any kangri-cancer or
erythema-ab-igne cases. Treat any output from this model as a research
demo, not a diagnostic tool.