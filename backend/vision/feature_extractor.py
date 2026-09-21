import numpy as np


class FeatureExtractor:

    def extract(self, hand_landmarks):
        """
        Convert 21 hand landmarks into a
        rotation-, translation-, and scale-normalized
        63-feature representation.

        21 landmarks × 3 coordinates = 63 features.
        """

        landmarks = []

        for landmark in hand_landmarks.landmark:

            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        return self.normalize_array(landmarks)

    @staticmethod
    def normalize_array(landmarks):
        """
        Canonicalize a 21 x 3 landmark array.
        """

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        ).reshape(21, 3)

        # -----------------------------------------
        # 1. Move wrist to origin
        # -----------------------------------------

        wrist = landmarks[0].copy()

        points = landmarks - wrist

        # -----------------------------------------
        # 2. Define a canonical palm coordinate system
        # -----------------------------------------

        # Middle-finger MCP gives the main palm direction
        y_axis = points[9].copy()

        y_length = np.linalg.norm(y_axis)

        if y_length < 1e-8:
            y_axis = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )
        else:
            y_axis = y_axis / y_length

        # Across-palm direction: index MCP -> pinky MCP
        x_axis = points[17] - points[5]

        # Remove the part parallel to y-axis
        x_axis = (
            x_axis
            - np.dot(x_axis, y_axis) * y_axis
        )

        x_length = np.linalg.norm(x_axis)

        if x_length < 1e-8:
            x_axis = np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32
            )
        else:
            x_axis = x_axis / x_length

        # Third axis
        z_axis = np.cross(
            x_axis,
            y_axis
        )

        z_length = np.linalg.norm(z_axis)

        if z_length < 1e-8:
            z_axis = np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32
            )
        else:
            z_axis = z_axis / z_length

        # Re-orthogonalize x-axis
        x_axis = np.cross(
            y_axis,
            z_axis
        )

        x_axis = x_axis / max(
            np.linalg.norm(x_axis),
            1e-8
        )

        # -----------------------------------------
        # 3. Rotate every landmark into canonical space
        # -----------------------------------------

        canonical = np.zeros_like(
            points
        )

        for i in range(21):

            point = points[i]

            canonical[i, 0] = np.dot(
                point,
                x_axis
            )

            canonical[i, 1] = np.dot(
                point,
                y_axis
            )

            canonical[i, 2] = np.dot(
                point,
                z_axis
            )

        # -----------------------------------------
        # 4. Normalize hand size
        # -----------------------------------------

        distances = np.linalg.norm(
            canonical,
            axis=1
        )

        scale = np.max(
            distances
        )

        if scale < 1e-8:
            scale = 1.0

        canonical = canonical / scale

        # -----------------------------------------
        # 5. Flatten to 63 features
        # -----------------------------------------

        return canonical.flatten().astype(
            np.float32
        )