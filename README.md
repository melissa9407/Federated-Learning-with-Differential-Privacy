# Federated-Learning-with-Differential-Privacy
Although the code is not exactly the same, this project was inspired by the innovative work done by [Yang, Y., Hui, B., Yuan, H., Gong, N., & Cao, Y. (2023). PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation. In Proceedings of the USENIX Security Symposium (Usenix'23)](https://github.com/BHui97/PrivateFL) which provided valuable insights into federated learning and differential privacy. We want to express our gratitude to the original authors and contributors of the project that inspired us for sharing their knowledge and resources with the open-source community.

This code is an implementation of a Federated Learning model with Differential Privacy. We use the methodology proposed by the authors of PrivateFL. The clients network structure includes a transformation layer that tackles client heterogeinity and preserves data features. The servers aggregates local models using FedAVG.
