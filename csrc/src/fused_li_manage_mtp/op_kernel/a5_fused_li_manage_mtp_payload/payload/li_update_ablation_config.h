#ifndef LI_UPDATE_ABLATION_CONFIG_H
#define LI_UPDATE_ABLATION_CONFIG_H

// TopK only carries token ids. Slots are resolved once after all speculative
// queries have been merged into a request-level union.
#define LI_UPDATE_ABLATION_MODE 0

#endif // LI_UPDATE_ABLATION_CONFIG_H
