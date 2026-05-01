const AUTH_KEY = "easydefect_authenticated_user";
let currentUserEmail = null;
let currentUserDisplayName = null;

// Remove any old persisted login from previous implementation.
try {
  localStorage.removeItem(AUTH_KEY);
} catch {
  // Ignore storage errors in restricted environments.
}

export const getCurrentUser = () => currentUserEmail;

/** Display name from signup/login, or fallback from email local-part. */
export const getCurrentUserDisplayName = () => {
  if (currentUserDisplayName && currentUserDisplayName.trim()) {
    return currentUserDisplayName.trim();
  }
  const em = currentUserEmail;
  if (em && em.includes("@")) {
    return em.split("@")[0].trim();
  }
  return em ? em.trim() : "";
};

export const isAuthenticated = () => Boolean(currentUserEmail);

export const loginUser = (email, displayName = "") => {
  currentUserEmail = email ? String(email).trim().toLowerCase() : null;
  currentUserDisplayName = displayName ? String(displayName).trim() : "";
  if (!currentUserDisplayName) {
    currentUserDisplayName = null;
  }
};

export const logoutUser = () => {
  currentUserEmail = null;
  currentUserDisplayName = null;
};

export const getAuthHeaders = () =>
  currentUserEmail ? { "X-User-Email": currentUserEmail } : {};
