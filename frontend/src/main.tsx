import { lazy, StrictMode, Suspense } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

// The admin dashboard is a separate chunk, so testers never download it.
const AdminApp = lazy(() => import('./admin/AdminApp'));
const isAdmin = window.location.pathname === '/admin' || window.location.pathname.startsWith('/admin/');

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isAdmin ? (
      <Suspense fallback={<p className="empty center">Loading…</p>}><AdminApp /></Suspense>
    ) : (
      <App />
    )}
  </StrictMode>,
);
